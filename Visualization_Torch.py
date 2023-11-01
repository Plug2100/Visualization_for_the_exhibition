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
predictions = []

# Initialize as white image
white_image = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255
camera_image = white_image
most_activated_filter_image = white_image
most_activated_filter_last_image = white_image
heatmap_image = white_image

app = Flask(__name__, template_folder='templates')


def construct_subclass_group_dict():
    labels = {}
    path_to_labels = 'Data/labels_de.txt'

    with open(path_to_labels, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, 0):
            labels[line_number] = line.strip()
    return labels


def generate_frames():
    # Load the labels
    class_labels = construct_subclass_group_dict()
    num_classes = len(class_labels)
    # Disable scientific notation for clarity

    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    layer_output = None

    def hook_fn(module, input, output):
        nonlocal layer_output
        layer_output = output

    layer_output_last = None

    def hook_fn_last(module, input, output):
        nonlocal layer_output_last
        layer_output_last = output

    # Load the model
    model = inceptionv1(pretrained=False)

    # Create a new fully connected layer
    new_fc = nn.Linear(in_features=model.softmax2_pre_activation_matmul.in_features, out_features=num_classes)
    # Replace the last layer of the pre-trained model with the new layer
    model.softmax2_pre_activation_matmul = new_fc
    model.load_state_dict(torch.load('model/pretrained_model_weights.pth'))

    model.eval()

    # List of hooks for max activations (first and last layer)
    desired_layer = model.mixed3a
    hook = desired_layer.register_forward_hook(hook_fn)
    desired_layer_last = model.mixed5b
    hook_last = desired_layer_last.register_forward_hook(hook_fn_last)

    # List of layers that are inputs to the pooling layer - may be can be done by calling model.mixed5b instead of all this
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
    camera = cv2.VideoCapture(0)
    while True:
        # Grab the webcam image
        ret, image = camera.read()
        input_tensor = preprocess(image)

        image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
        # Add a batch dimension (GoogleNet expects batched input)
        input_tensor = input_tensor.unsqueeze(0)
        with torch.no_grad():
            prediction = model(input_tensor)

        # Find the most activated filter in the layer
        most_activated_filter = layer_output.mean(dim=(0, 2, 3)).argmax()
        most_activated_filter = int(most_activated_filter)
        most_act_layer = cv2.imread(f'visualisations/400/mixed3a/{most_activated_filter}.jpg')
        most_act_layer = cv2.resize(most_act_layer, (image_size, image_size))

        # Find the most activated filter in the layer last
        most_activated_filter_last = layer_output_last.mean(dim=(0, 2, 3)).argmax()
        most_activated_filter_last = int(most_activated_filter_last)
        most_act_layer_last = cv2.imread(f'visualisations/400/mixed5b/{most_activated_filter_last}.jpg')
        most_act_layer_last = cv2.resize(most_act_layer_last, (image_size, image_size))

        # Concat activations of conv layers for the heatmap
        concat_cn = torch.cat((activation['1st'], activation['2nd'], activation['3rd'], activation['4th']), dim=1)

        # Receive answer from the model
        predicted_class = torch.argmax(prediction, dim=1).item()
        top3 = torch.topk(prediction.flatten(), 3).indices
        class_name_1 = class_labels[top3[0].item()]
        class_name_2 = class_labels[top3[1].item()]
        class_name_3 = class_labels[top3[2].item()]
        prediction_score_1 = prediction[0][top3[0].item()]
        prediction_score_2 = prediction[0][top3[1].item()]
        prediction_score_3 = prediction[0][top3[2].item()]

        # Building Heatmap
        result_heatmap = concat_cn * model.softmax2_pre_activation_matmul.weight.data[predicted_class].view(1, 1024, 1, 1)  # 1024 - input to the last FC layer
        result_heatmap = result_heatmap.squeeze()
        result_heatmap = torch.relu(result_heatmap) / torch.max(result_heatmap)  # Normalisation
        result_heatmap = result_heatmap.numpy()
        result_heatmap = np.sum(result_heatmap, axis=0)
        result_heatmap = cv2.resize(result_heatmap, (image_size, image_size))
        result_heatmap = result_heatmap * bright  # make heatmap more bright
        heatmap_colormap = cv2.applyColorMap(np.uint8(result_heatmap), cv2.COLORMAP_JET)

        # Combine Heatmap and Actual Image
        combined_image = cv2.addWeighted(image, 1 - alpha, heatmap_colormap, alpha, 0)

        # Save predictions and images in global variables, so they can be fetched with javascript
        results = [f"{class_name_1} | {str(np.round(prediction_score_1.item() * 100, decimals=2))}%",
                   f"{class_name_2} | {str(np.round(prediction_score_2.item() * 100, decimals=2))}%",
                   f"{class_name_3} | {str(np.round(prediction_score_3.item() * 100, decimals=2))}%"]
        set_predictions(results)
        set_images(image, most_act_layer, most_act_layer_last, combined_image)

        # Return webcam image
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def set_predictions(results):
    global predictions
    predictions = results


def set_images(image, maf_image, mafl_image, heatmap):
    global camera_image
    camera_image = image
    global most_activated_filter_image
    most_activated_filter_image = maf_image
    global most_activated_filter_last_image
    most_activated_filter_last_image = mafl_image
    global heatmap_image
    heatmap_image = heatmap


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_predictions', methods=['GET'])
def get_predictions():
    return jsonify(data=predictions)


@app.route('/get_camera_image', methods=['GET'])
def get_camera_image():
    ret, buffer = cv2.imencode('.jpg', camera_image)
    frame = buffer.tobytes()
    return Response(frame, content_type='image/jpeg')


@app.route('/get_maf_image', methods=['GET'])
def get_most_activated_filter_image():
    ret, buffer = cv2.imencode('.jpg', most_activated_filter_image)
    frame = buffer.tobytes()
    return Response(frame, content_type='image/jpeg')


@app.route('/get_mafl_image', methods=['GET'])
def get_most_activated_filter_last_image():
    ret, buffer = cv2.imencode('.jpg', most_activated_filter_last_image)
    frame = buffer.tobytes()
    return Response(frame, content_type='image/jpeg')


@app.route('/get_heatmap_image', methods=['GET'])
def get_heatmap_image():
    ret, buffer = cv2.imencode('.jpg', heatmap_image)
    frame = buffer.tobytes()
    return Response(frame, content_type='image/jpeg')


if __name__ == '__main__':
    app.run(debug=True)
