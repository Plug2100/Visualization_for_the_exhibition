import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from flask import Flask, Response, render_template, jsonify
from lucent.modelzoo import inceptionv1

num_classes = 15
image_size = 720  # Size of images (original, heatmap and act.layers)
bright = 3.0  # How bright is the Heatmap
alpha = 0.5  # Adjust the alpha value to control the intensity of the heatmap overlay
layers = ['mixed3a', 'mixed4b', 'mixed5b']
num_filters_per_layer = 3
predictions = {}
white_image = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255
imageTypes = ['camera', 'heatmap', 'mixed3a_1', 'mixed3a_2', 'mixed3a_3', 'mixed4b_1', 'mixed4b_2', 'mixed4b_3', 'mixed5b_1', 'mixed5b_2', 'mixed5b_3']
global_images = {imgType: white_image for imgType in imageTypes}

app = Flask(__name__, template_folder='templates')


def construct_subclass_group_dict(lang):
    labels = {}
    path_to_labels = 'Data/labels_de.txt' if lang == 'de' else 'Data/labels_en.txt'

    with open(path_to_labels, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, 0):
            labels[line_number] = line.strip()
    return labels


def generate_frames():
    # Load the labels
    class_labels_de = construct_subclass_group_dict('de')
    class_labels_en = construct_subclass_group_dict('en')

    num_classes = len(class_labels_de)
    # Disable scientific notation for clarity

    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    # Load the model
    model = inceptionv1(pretrained=False)

    # Create a new fully connected layer
    new_fc = nn.Linear(in_features=model.softmax2_pre_activation_matmul.in_features, out_features=num_classes)
    # Replace the last layer of the pre-trained model with the new layer
    model.softmax2_pre_activation_matmul = new_fc
    model.load_state_dict(torch.load('model/pretrained_model_weights.pth'))

    model.eval()

    # Set hooks for intermediate layer visualization
    model.mixed3a.register_forward_hook(get_activation('mixed3a'))
    model.mixed4b.register_forward_hook(get_activation('mixed4b'))
    model.mixed5b.register_forward_hook(get_activation('mixed5b'))

    # List of layers that are inputs to the pooling layer - may be can be done by calling model.mixed5b instead of all this
    # Set hooks for class activation mapping (CAM) of the last layer
    model.mixed5b_1x1_pre_relu_conv.register_forward_hook(get_activation('1st'))
    model.mixed5b_3x3_pre_relu_conv.register_forward_hook(get_activation('2nd'))
    model.mixed5b_5x5_pre_relu_conv.register_forward_hook(get_activation('3rd'))
    model.mixed5b_pool_reduce_pre_relu_conv.register_forward_hook(get_activation('4th'))

    # Define the image preprocessing transformations
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print(torch.__version__)
    # CAMERA 
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        # Grab the webcam image
        ret, image = camera.read()
        input_tensor = preprocess(image)

        image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
        images = {'camera': image}

        # Add a batch dimension (GoogleNet expects batched input)
        input_tensor = input_tensor.unsqueeze(0)
        with torch.no_grad():
            prediction = model(input_tensor)

        def get_nth_most_activated_filter(layer, n):
            # Find the n-th most activated filter in the layer
            activations = activation[layer].mean(dim=(0, 2, 3))
            indices = np.argpartition(activations, kth=-n)
            filter_image = cv2.imread(f'visualisations/400/{layer}/{indices[-n]}.jpg')
            filter_image = cv2.resize(filter_image, (image_size, image_size))
            return filter_image

        # Get most activated filters
        for layer in layers:
            for i in range(1, num_filters_per_layer + 1):
                images[f'{layer}_{i}'] = get_nth_most_activated_filter(layer, i)

        # Concat activations of conv layers for the heatmap
        concat_cn = torch.cat((activation['1st'], activation['2nd'], activation['3rd'], activation['4th']), dim=1)

        # Get predictions from model
        predicted_class = torch.argmax(prediction, dim=1).item()
        top3 = torch.topk(prediction.flatten(), 3).indices

        # Building heatmap
        result_heatmap = concat_cn * model.softmax2_pre_activation_matmul.weight.data[predicted_class].view(1, 1024, 1, 1)  # 1024 - input to the last FC layer
        result_heatmap = result_heatmap.squeeze()
        result_heatmap = torch.relu(result_heatmap) / torch.max(result_heatmap)  # Normalisation
        result_heatmap = result_heatmap.numpy()
        result_heatmap = np.sum(result_heatmap, axis=0)
        result_heatmap = cv2.resize(result_heatmap, (image_size, image_size))
        result_heatmap = result_heatmap * bright  # make heatmap more bright
        heatmap_colormap = cv2.applyColorMap(np.uint8(result_heatmap), cv2.COLORMAP_JET)

        # Combine heatmap and actual image
        combined_image = cv2.addWeighted(image, 1 - alpha, heatmap_colormap, alpha, 0)
        images['heatmap'] = combined_image

        # Save predictions and images in global variables, so they can be fetched with javascript
        prediction_scores = [f"{str(np.round(prediction[0][top3[0].item()].item() * 100, decimals=2))}%",
                             f"{str(np.round(prediction[0][top3[1].item()].item() * 100, decimals=2))}%",
                             f"{str(np.round(prediction[0][top3[2].item()].item() * 100, decimals=2))}%"]
        predicted_classes = [f"{class_labels_de[top3[0].item()]} <br> ({class_labels_en[top3[0].item()]})",
                             f"{class_labels_de[top3[1].item()]} <br> ({class_labels_en[top3[1].item()]})",
                             f"{class_labels_de[top3[2].item()]} <br> ({class_labels_en[top3[2].item()]})"]
        prediction_dict = {"scores": prediction_scores, "classes": predicted_classes}
        set_predictions(prediction_dict)
        set_images(images)

        # Return webcam image
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def set_predictions(results):
    global predictions
    predictions = results


def set_images(images):
    global global_images
    for key in images.keys():
        global_images[key] = images[key]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_predictions', methods=['GET'])
def get_predictions():
    return jsonify(data=predictions)


@app.route('/get_image/<key>', methods=['GET'])
def get_image(key):
    ret, buffer = cv2.imencode('.jpg', global_images[key])
    frame = buffer.tobytes()
    return Response(frame, content_type='image/jpeg')


if __name__ == '__main__':
    app.run(debug=True)
