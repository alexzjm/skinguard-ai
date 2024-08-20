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
    console.log("Testing script.js");

    const fileInput = document.getElementById('file');
    const preview = document.getElementById('preview');

    fileInput.addEventListener('change', function (event) {
        const file = fileInput.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function (e) {
                preview.src = e.target.result;
                preview.style.display = 'block';
            }
            reader.readAsDataURL(file);
        }
    });

    document.getElementById('uploadForm').addEventListener('submit', function (event) {  // Changed form id
        event.preventDefault();
        const formData = new FormData();
        const fileInput = document.getElementById('file');  // Changed element id to file
        formData.append('file', fileInput.files[0]);
        formData.append('model', currentModelId);

        fetch('/api/upload', {
            method: 'POST',
            body: formData
        })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    document.getElementById('result').innerText = 'Error: ' + data.error;
                } else {
                    if (data.model_used == 0){ //processing based on the model used
                        threshold = 0.5;
                        if (data.prediction <= threshold) {
                            document.getElementById('result').innerText = "You are cancer free. Score: " + data.prediction;
                        } else {
                            document.getElementById('result').innerText = "Please see a professional for further diagnosis. Percent: " + data.prediction * 50 + ". Score: " + data.prediction;
                        }
                    }

                }
            })
            .catch(error => console.error('Error:', error));
    });
});