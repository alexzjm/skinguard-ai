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

    // Image preview functionality
    fileInput.addEventListener('change', function (event) {
        const file = fileInput.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function (e) {
                preview.src = e.target.result;
                preview.style.display = 'block';
            }
            reader.readAsDataURL(file);
            
            // Clear previous results
            resultDiv.innerHTML = '';
        }
    });

    // Form submission
    document.getElementById('uploadForm').addEventListener('submit', async function (event) {
        event.preventDefault();
        
        const file = fileInput.files[0];
        if (!file) {
            resultDiv.innerHTML = '<p class="error">Please select an image first.</p>';
            return;
        }

        // Show loading state
        resultDiv.innerHTML = '<p class="loading">Analyzing image...</p>';

        try {
            const formData = new FormData();
            formData.append('file', file);
            formData.append('model', '0'); // Always use the only working model

            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.error) {
                resultDiv.innerHTML = `<p class="error">Error: ${data.error}</p>`;
            } else {
                displayResult(data.prediction);
            }
        } catch (error) {
            console.error('Error:', error);
            resultDiv.innerHTML = '<p class="error">An error occurred while processing your image. Please try again.</p>';
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