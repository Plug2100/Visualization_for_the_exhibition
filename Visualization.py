from keras.models import load_model  # TensorFlow is required for Keras to work
import keras
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from flask import Flask, Response, render_template
import seaborn as sns


app = Flask(__name__, template_folder='templates')


def model_description(model):
    print('DESCRIPTION OF THE MODEL:')
    model.summary()
    # Access the layers of the first sequential model (sequential_1)
    sequential_1_layers = model.layers[0].layers
    # Access the layers of the second sequential model (sequential_3)
    sequential_3_layers = model.layers[1].layers
    # Print the layers of the first sequential model (sequential_1)
    print("Layers of sequential_1:")
    for layer in sequential_1_layers:
        print(layer.name)
        if(layer.name == 'model1'):
            print('Consist of:')
            layer.summary()
    #Print the layers of the second sequential model (sequential_3)
    print("\nLayers of sequential_3:")
    for layer in sequential_3_layers:
       print(layer.name)


def by_photo():
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)
    # Load the model
    model = load_model("converted_keras/keras_Model.h5", compile=False)
    # Load the labels
    class_names = open("converted_keras/labels.txt", "r").readlines()
    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    # Replace this with the path to your image
    image = Image.open("Data/Dog.jpg").convert("RGB")
    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    # turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    # Load the image into the array
    data[0] = normalized_image_array
    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", confidence_score)
    result_heatmap = CAM(model, size, data, index)
    plt.imshow(image)
    sns.heatmap(result_heatmap, cmap='jet', alpha=0.5)
    plt.show()



def CAM(model, size, data, index):
    # A new modelg to extract map. If will affect the performance - somehow extract info from model during 1st inference
    model_for_map_extraction = tf.keras.models.Model(inputs=model.get_layer('sequential_1').get_layer('model1').input, outputs= [model.get_layer('sequential_1').get_layer('model1').get_layer('out_relu').output])
    map_for_heatmap = model_for_map_extraction(data)
    layer_name = 'dense_Dense2'  # Last Dence layer out of 2
    weights = model.get_layer('sequential_3').get_layer(layer_name).get_weights()
    column_of_won_result = weights[0][:, index] # extract weights for correct label
    layer_name = 'dense_Dense1'  #  First Dence layer out of 2
    weights = model.get_layer('sequential_3').get_layer(layer_name).get_weights()[0]
    weights_for_heatmap = np.dot(weights, column_of_won_result)
    result_heatmap = map_for_heatmap * weights_for_heatmap
    result_heatmap = tf.squeeze(result_heatmap)
    result_heatmap = tf.maximum(result_heatmap, 0) / tf.math.reduce_max(result_heatmap) # Normalisation
    result_heatmap = result_heatmap.numpy()
    result_heatmap = np.sum(result_heatmap, axis=2)
    result_heatmap = cv2.resize(result_heatmap, (size[0], size[1]))
    return (result_heatmap) 



def generate_frames():

    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)
    # Load the model
    model = load_model("converted_keras/keras_Model.h5", compile=False)
    # Load the labels
    class_names = open("converted_keras/labels.txt", "r").readlines()
    # CAMERA can be 0 or 1 based on default camera of your computer

    camera = cv2.VideoCapture(0)
    while True:
        # Grab the webcamera's image.
        ret, image = camera.read()
        # Resize the raw image into (224-height,224-width) pixels
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        cv2.imwrite("Data/WebCam.jpg", image)
        # Show the image in a window
        image_backup = image
        # Make the image a numpy array and reshape it to the models input shape.
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
        # Normalize the image array
        image = (image / 127.5) - 1
        # Predicts the model
        prediction = model.predict(image)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        result_heatmap = CAM(model, (224, 224), image, index)
        result_heatmap = result_heatmap * 3.0 # make heatmap more bright
        # print("Class:", class_name[2:], end="")
        # print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
        heatmap_colormap = cv2.applyColorMap(np.uint8(result_heatmap), cv2.COLORMAP_JET)
        alpha = 0.5  # Adjust the alpha value to control the intensity of the heatmap overlay
        combined_image = cv2.addWeighted(image_backup, 1 - alpha, heatmap_colormap, alpha, 0)
        combined_image = cv2.resize(combined_image, (300, 300))
        image_backup = cv2.resize(image_backup, (300, 300))

        white_image = np.ones((300, 200, 3), dtype=np.uint8) * 255  
        # Concatenate the images horizontally
        final_image = np.hstack((image_backup, white_image, combined_image))
        # Concatenate the images verticaly (to put the text)
        white_image = np.ones((100, 800, 3), dtype=np.uint8) * 255 
        final_image = np.vstack((final_image, white_image))
        text = f"Class: {class_name[2:-1]} | Confidence: {str(np.round(confidence_score * 100))[:-2]}%"
        cv2.putText(final_image, text, (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        ret, buffer = cv2.imencode('.jpg', final_image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed1')
def video_feed1():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)


