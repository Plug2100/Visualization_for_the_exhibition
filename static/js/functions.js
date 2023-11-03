function updatePredictions() {
    // Make an AJAX request to fetch updated data
    fetch('/get_predictions')
        .then(response => response.json())
        .then(data => {
            // Update the HTML content with the fetched data
            document.getElementById('prediction-score-1').innerHTML = data.data.scores[0];
            document.getElementById('prediction-score-2').innerHTML = data.data.scores[1];
            document.getElementById('prediction-score-3').innerHTML = data.data.scores[2];
            document.getElementById('prediction-class-1').innerHTML = data.data.classes[0];
            document.getElementById('prediction-class-2').innerHTML = data.data.classes[1];
            document.getElementById('prediction-class-3').innerHTML = data.data.classes[2];
        })
        .catch(error => {
            console.error('Error fetching data:', error);
        });
}

function runWebcam() {
    // Make an AJAX request to fetch updated data
    fetch('/video_feed')
        .then(response => response.blob())
        .then(blob => {
        })
        .catch(error => {
            console.error('Error fetching video_feed:', error);
        });
}

function updateImages() {
    const imageTypes = ['camera', 'heatmap', 'mixed3a_1', 'mixed3a_2', 'mixed3a_3', 'mixed4b_1', 'mixed4b_2', 'mixed4b_3', 'mixed5b_1', 'mixed5b_2', 'mixed5b_3'];
    for (let i = 0; i < imageTypes.length; i++) {
        const imageType = imageTypes[i];

        // Make an AJAX request to fetch updated data
        fetch('/get_image/' + imageType)
            .then(response => response.blob())
            .then(blob => {
                // Create a data URL from the image data
                const url = URL.createObjectURL(blob);

                // Display the image
                const imgElement = document.getElementById(imageType + '_image');
                imgElement.src = url;
            })
            .catch(error => {
                console.error('Error fetching ' + imageType + '_image:', error);
            });
    }
}

function updateContent() {
    updatePredictions();
    updateImages();
}

// Calls the generate_frames() method which will run endlessly
runWebcam()

// Periodically refresh content
setInterval(updateContent, 5000);