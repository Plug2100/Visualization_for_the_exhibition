function updatePredictions() {
    // Make an AJAX request to fetch updated data
    fetch('/get_predictions')
        .then(response => response.json())
        .then(data => {
            // Update the HTML content with the fetched data
            document.getElementById('prediction-1').textContent = data.data[0];
            document.getElementById('prediction-2').textContent = data.data[1];
            document.getElementById('prediction-3').textContent = data.data[2];
        })
        .catch(error => {
            console.error('Error fetching data:', error);
        });
}

function updateImages() {
    const imageTypes = ['camera', 'maf', 'mafl', 'heatmap'];
    for (let i = 0; i < imageTypes.length; i++) {
        const imageType = imageTypes[i];

        // Make an AJAX request to fetch updated data
        fetch('/get_' + imageType + '_image')
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

// Periodically refresh content
setInterval(updateContent, 5000);