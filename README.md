# Neural Storyteller App

This is a web application that generates creative captions (stories) for uploaded images using a deep learning model.

## Project Structure

- `backend/server.py`: FastAPI server that loads the PyTorch model and serves the prediction endpoint.
- `frontend/`: Contains the web interface (HTML, CSS, JS).
- `model.pth`: The trained model weights.
- `captions.txt`: The caption dataset used for vocabulary building.

## Prerequisites

- Python 3.8+
- PyTorch
- FastAPI
- Uvicorn
- Pandas
- Pillow

## How to Run

1.  **Start the Backend Server:**
    Open a terminal in the project root directory and run:
    ```bash
    uivcorn backend.server:app --reload --port 8000
    ```
    Wait for the message: `Application startup complete.`

2.  **Launch the Frontend:**
    You have two options:
    - **Option A (Simple server):** run `python -m http.server 3000` inside the `frontend` folder, then open `http://localhost:3000` in your browser.
    - **Option B (Direct file):** Simply open `frontend/index.html` in your browser.

3.  **Use the App:**
    - Upload an image.
    - Click "Tell Story".
    - Wait for result.

## Notes

- The first run might take a while to download ResNet50 weights if not cached.
- Ensure `model.pth` and `captions.txt` are present in the root directory.
