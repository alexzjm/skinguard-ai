let currentModelId = "";
const allModels = ["Skin Cancer AI", "Model 2", "Model 3"]
const allDescriptions = [
    "Example description 1 " +
    "through all these lines", 

    "Example description 2", 

    "Example description 3"
]

function switchPage(elementId){
    console.log(elementId);
    currentModelId = elementId;
    document.getElementById(elementId).style.display = "block";
    document.getElementById("about-us-div").style.display = "none";
}

function navigatePage(elementId){
    document.getElementById("ai-classification-div").style.display = "none";
    document.getElementById("result").innerHTML = "";
    if (elementId == -1){
        document.getElementById("home-page").style.display = "block";
        document.getElementById("instructions").style.display = "none";
        document.getElementById("terms-and-conditions").style.display = "none";
        document.getElementById("model-name").innerHTML = "";
        currentModelId = "";
    } else {
        document.getElementById("home-page").style.display = "none";
        document.getElementById("instructions").style.display = "block";
        document.getElementById("terms-and-conditions").style.display = "flex";
        document.getElementById("model-name").innerHTML = allModels[elementId];
        document.getElementById("model-description").innerHTML = allDescriptions[elementId];
        currentModelId = elementId;
        console.log(currentModelId);
    }
}

function agreeToTerms(){
    document.getElementById("terms-and-conditions").style.display = "none";
    document.getElementById("ai-classification-div").style.display = "block";
}

/*function navigatePage(elementId){
    const pages = ["home-page", "skin-cancer-ai"];
    for (let i = 0; i < pages.length; i++){
        if (pages[i] == elementId){
            document.getElementById(elementId).style.display = "block";
        } else {
            document.getElementById(pages[i]).style.display = "none";
        }
    }
}*/

document.addEventListener('DOMContentLoaded', function () {
    const fileInput = document.getElementById('file');
    const preview = document.getElementById('preview');
    const resultDiv = document.getElementById('result');

    // File validation configuration
    const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
    const ALLOWED_TYPES = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp', 'image/webp'];
    const MIN_FILE_SIZE = 1024; // 1KB minimum

    // Image preview and validation functionality
    fileInput.addEventListener('change', function (event) {
        const file = fileInput.files[0];
        
        // Clear previous results and preview
        resultDiv.innerHTML = '';
        preview.style.display = 'none';
        preview.src = '';
        
        if (file) {
            // Validate file
            const validationResult = validateFile(file);
            if (!validationResult.isValid) {
                showError(validationResult.message);
                fileInput.value = ''; // Clear the input
                return;
            }
            
            // Show preview if validation passes
            const reader = new FileReader();
            reader.onload = function (e) {
                preview.src = e.target.result;
                preview.style.display = 'block';
            }
            reader.readAsDataURL(file);
        }
    });

    // File validation function
    function validateFile(file) {
        // Check if file exists
        if (!file) {
            return { isValid: false, message: 'No file selected.' };
        }
        
        // Check file size (too large)
        if (file.size > MAX_FILE_SIZE) {
            return { 
                isValid: false, 
                message: `File too large. Maximum size is ${MAX_FILE_SIZE / (1024 * 1024)}MB. Your file is ${(file.size / (1024 * 1024)).toFixed(1)}MB.` 
            };
        }
        
        // Check file size (too small)
        if (file.size < MIN_FILE_SIZE) {
            return { 
                isValid: false, 
                message: 'File too small. Please select a valid image file.' 
            };
        }
        
        // Check file type
        if (!ALLOWED_TYPES.includes(file.type)) {
            return { 
                isValid: false, 
                message: `Invalid file type. Please upload: JPEG, PNG, GIF, BMP, or WebP images. You uploaded: ${file.type || 'unknown type'}` 
            };
        }
        
        return { isValid: true, message: '' };
    }

    // Error display function
    function showError(message) {
        resultDiv.innerHTML = `<p class="error">${message}</p>`;
    }

    // Success message function
    function showSuccess(message) {
        resultDiv.innerHTML = `<p class="success">${message}</p>`;
    }

    // Form submission with enhanced validation
    document.getElementById('uploadForm').addEventListener('submit', async function (event) {
        event.preventDefault();
        
        const file = fileInput.files[0];
        
        // Pre-submission validation
        if (!file) {
            showError('Please select an image first.');
            return;
        }

        // Re-validate file before submission (in case file was changed)
        const validationResult = validateFile(file);
        if (!validationResult.isValid) {
            showError(validationResult.message);
            fileInput.value = '';
            preview.style.display = 'none';
            return;
        }

        // Disable submit button to prevent double submission
        const submitButton = event.target.querySelector('button[type="submit"]');
        const originalButtonText = submitButton.textContent;
        submitButton.disabled = true;
        submitButton.textContent = 'Analyzing...';

        // Show loading state
        resultDiv.innerHTML = '<p class="loading">Analyzing image, please wait...</p>';

        try {
            const formData = new FormData();
            formData.append('file', file);
            formData.append('model', '0'); // Always use the only working model

            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            // Check if response is ok
            if (!response.ok) {
                throw new Error(`Server error: ${response.status} ${response.statusText}`);
            }

            const data = await response.json();

            if (data.error) {
                showError(`Analysis failed: ${data.error}`);
            } else if (data.prediction !== undefined) {
                displayResult(data.prediction);
            } else {
                showError('Invalid response from server. Please try again.');
            }
        } catch (error) {
            console.error('Upload error:', error);
            
            // Provide specific error messages based on error type
            if (error.name === 'TypeError' && error.message.includes('fetch')) {
                showError('Network error. Please check your internet connection and try again.');
            } else if (error.message.includes('Server error: 413')) {
                showError('File too large for server. Please try a smaller image.');
            } else if (error.message.includes('Server error: 500')) {
                showError('Server error occurred. Please try again later.');
            } else {
                showError('An unexpected error occurred while processing your image. Please try again.');
            }
        } finally {
            // Re-enable submit button
            submitButton.disabled = false;
            submitButton.textContent = originalButtonText;
        }
    });

    function displayResult(prediction) {
        const threshold = 0.5;
        const percentage = (prediction * 100).toFixed(1);
        
        if (prediction <= threshold) {
            resultDiv.innerHTML = `
                <div class="result-success">
                    <h3>Low Risk Detected</h3>
                    <p>Confidence Score: ${percentage}%</p>
                    <p>The analysis suggests a low probability of malignancy.</p>
                </div>
            `;
        } else {
            resultDiv.innerHTML = `
                <div class="result-warning">
                    <h3>High Risk Detected</h3>
                    <p>Confidence Score: ${percentage}%</p>
                    <p><strong>Please consult a healthcare professional for further evaluation.</strong></p>
                </div>
            `;
        }
    }
});