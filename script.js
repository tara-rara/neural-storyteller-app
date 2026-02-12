const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const uploadContent = document.getElementById('upload-content');
const previewContainer = document.getElementById('preview-container');
const imagePreview = document.getElementById('image-preview');
const removeBtn = document.getElementById('remove-btn');
const generateBtn = document.getElementById('generate-btn');
const resultContent = document.getElementById('result-content');
const loadingSpinner = document.getElementById('loading-spinner');

let currentFile = null;

// Drag & Drop
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
    }
});

dropZone.addEventListener('click', () => {
    if (!currentFile) {
        fileInput.click();
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert("Please upload an image file.");
        return;
    }
    
    currentFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        previewContainer.classList.remove('hidden');
        uploadContent.style.opacity = '0';
        generateBtn.disabled = false;
        
        // Clear previous results
        resetResult();
    };
    reader.readAsDataURL(file);
}

removeBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    currentFile = null;
    fileInput.value = '';
    imagePreview.src = '';
    previewContainer.classList.add('hidden');
    uploadContent.style.opacity = '1';
    generateBtn.disabled = true;
    resetResult();
});

function resetResult() {
    resultContent.innerHTML = '<p class="placeholder-text">Your story will appear here...</p>';
    loadingSpinner.classList.add('hidden');
}

// Generate Caption
generateBtn.addEventListener('click', async () => {
    if (!currentFile) return;

    // Show Loading
    resultContent.innerHTML = '';
    resultContent.appendChild(loadingSpinner);
    loadingSpinner.classList.remove('hidden');
    generateBtn.disabled = true;

    const formData = new FormData();
    formData.append('file', currentFile);

    try {
        const response = await fetch('http://localhost:8000/predict', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Failed to generate story');
        }

        const data = await response.json();
        const caption = data.caption;
        
        // Display Result with Typewriter effect
        displayResult(caption);

    } catch (error) {
        console.error(error);
        resultContent.innerHTML = `<p style="color: #ff6b6b;">Error: ${error.message}</p>`;
    } finally {
        generateBtn.disabled = false;
    }
});

function displayResult(text) {
    resultContent.innerHTML = '';
    const p = document.createElement('p');
    p.classList.add('typing-cursor');
    p.style.fontSize = '1.2rem';
    p.style.lineHeight = '1.6';
    p.style.color = '#fff';
    resultContent.appendChild(p);

    let i = 0;
    const speed = 50; // ms per char

    function typeWriter() {
        if (i < text.length) {
            p.textContent += text.charAt(i);
            i++;
            setTimeout(typeWriter, speed);
        } else {
            p.classList.remove('typing-cursor');
        }
    }
    
    typeWriter();
}
