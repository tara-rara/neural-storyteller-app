import logging
import os
import pickle
import re
from collections import Counter
from io import BytesIO

import kagglehub
import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import models, transforms

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
MODEL_PATH = "model.pth"
CAPTIONS_FILE = "captions.txt"  # Will attempt to find/download via kagglehub
EMBED_SIZE = 512
HIDDEN_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MODEL DEFINITIONS (COPIED FROM NOTEBOOK) ---

class Encoder(nn.Module):
    def __init__(self, embed_size):
        super(Encoder, self).__init__()
        # Projects 2048-dim ResNet features
        self.fc = nn.Linear(2048, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, features):
        return self.dropout(self.relu(self.fc(features)))

class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions[:, :-1]))
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs

class NeuralStoryteller(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(NeuralStoryteller, self).__init__()
        self.encoder = Encoder(embed_size)
        self.decoder = Decoder(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

# --- VOCABULARY ---

class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.stoi = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer(text):
        text = str(text).lower()
        text = re.sub(r'[^a-z\s]', '', text)
        return text.split()

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4
        for sentence in sentence_list:
            for word in self.tokenizer(sentence):
                frequencies[word] += 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer(text)
        return [self.stoi.get(token, self.stoi["<unk>"]) for token in tokenized_text]

# --- APPLICATION ---

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
vocab = None
feature_extractor = None  # ResNet50
transform = None

def load_resources():
    global model, vocab, feature_extractor, transform
    
    # 1. Load/Find Dataset for Vocab
    logger.info("Loading resource dependencies...")
    
    try:
        if os.path.exists("captions.txt"):
            logger.info("Found local captions.txt, using it.")
            captions_path = "captions.txt"
        else:
            # Attempt to use cached vocab if exists (not implemented here but good practice)
            # Instead, fetch dataset via KaggleHub
            logger.info("Attempting to download dataset via kagglehub...")
            path = kagglehub.dataset_download("adityajn105/flickr30k")
            captions_path = os.path.join(path, "captions.txt")
        
        if not os.path.exists(captions_path):
            raise FileNotFoundError(f"Could not find captions.txt at {captions_path}")
            
        logger.info(f"Using captions from: {captions_path}")
        df = pd.read_csv(captions_path)
        caption_col = 'comment' if 'comment' in df.columns else df.columns[-1]
        
        # Build Vocab
        logger.info("Building vocabulary...")
        vocab = Vocabulary(freq_threshold=5)
        vocab.build_vocabulary(df[caption_col].tolist())
        logger.info(f"Vocabulary size: {len(vocab)}")
        
    except Exception as e:
        logger.error(f"Failed to load dataset/vocab: {e}")
        # Build dummy vocab if strictly necessary (will produce bad output) but better to fail.
        raise e

    # 2. Setup Feature Extractor (ResNet50)
    logger.info("Setting up ResNet50 feature extractor...")
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    modules = list(resnet.children())[:-1]  # Remove classification layer
    feature_extractor = nn.Sequential(*modules).to(DEVICE)
    feature_extractor.eval()
    
    # 3. Setup Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # 4. Load Model
    logger.info(f"Loading model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Key file {MODEL_PATH} is missing!")
        
    m = NeuralStoryteller(EMBED_SIZE, HIDDEN_SIZE, len(vocab)).to(DEVICE)
    
    # Load state dict
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    
    # Handle module. prefix if present (from DataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    m.load_state_dict(new_state_dict)
    m.eval()
    model = m
    logger.info("Model loaded successfully!")

@app.on_event("startup")
async def startup_event():
    load_resources()

def greedy_search_inference(image_tensor):
    model.eval()
    result_caption = []
    
    with torch.no_grad():
        # Feature extraction
        # image_tensor shape: (1, 3, 224, 224)
        features = feature_extractor(image_tensor) # (1, 2048, 1, 1)
        features = features.view(features.size(0), -1) # (1, 2048)
        
        # Encoder
        features = model.encoder(features).unsqueeze(1) # (1, 1, embed_size)
        
        input_word = torch.tensor([vocab.stoi["<start>"]]).to(DEVICE).unsqueeze(0)
        states = None
        
        for i in range(20): # max_len
            embeddings = model.decoder.embed(input_word)
            
            if i == 0:
                embeddings = torch.cat((features, embeddings), dim=1)
            
            hiddens, states = model.decoder.lstm(embeddings, states)
            outputs = model.decoder.linear(hiddens[:, -1, :])
            predicted = outputs.argmax(dim=1)
            
            word = vocab.itos[predicted.item()]
            
            if word == "<end>":
                break
                
            result_caption.append(word)
            input_word = predicted.unsqueeze(0)
            
    return " ".join(result_caption)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not model:
        return JSONResponse(status_code=503, content={"error": "Model not loaded yet"})
    
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        
        # Preprocess
        tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        # Predict
        caption = greedy_search_inference(tensor)
        
        return {"caption": caption}
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
