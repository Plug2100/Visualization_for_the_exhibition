import numpy as np
import torch
import cv2
from flask import Flask, Response, render_template
import torchvision.transforms as transforms
import torch
import time

app = Flask(__name__, template_folder='templates')



def generate_frames():
    # Disable scientific notation for clarity
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    np.set_printoptions(suppress=True)
    # Load the model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)
    model.eval()

    # List of layers that are inputs to the pooling layer
    model.inception5b.branch1.conv.register_forward_hook(get_activation('1st'))
    model.inception5b.branch2[1].conv.register_forward_hook(get_activation('2nd'))
    model.inception5b.branch3[1].conv.register_forward_hook(get_activation('3rd'))
    model.inception5b.branch4[1].conv.register_forward_hook(get_activation('4th'))

    # Define the image preprocessing transformations
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #TODO - Later change to the other type of norm
    ])
    # Load the labels
    class_labels = []
    with open("Data/imagenet-classes.txt") as f:
        class_labels = [line.strip() for line in f.readlines()]

    # CAMERA can be 0 or 1 based on default camera of your computer
    camera = cv2.VideoCapture(0)
    while True:
        # Grab the webcamera's image.
        ret, image = camera.read()
        input_tensor = preprocess(image)
        image = cv2.resize(image, (720, 720), interpolation=cv2.INTER_AREA)

        # Add a batch dimension (GoogleNet expects batched input)
        input_tensor = input_tensor.unsqueeze(0)
        with torch.no_grad():
            prediction = model(input_tensor)

        concat_cn = torch.cat((activation['1st'], activation['2nd'], activation['3rd'], activation['4th']), dim=1)

        predicted_class = torch.argmax(prediction, dim=1).item()
        class_name = class_labels[predicted_class]
        confidence_score = torch.nn.functional.softmax(prediction[0], dim=0)
        confidence_score = confidence_score[predicted_class]
        result_heatmap = concat_cn * model.fc.weight.data[predicted_class].view(1, 1024, 1, 1)
        result_heatmap = result_heatmap.squeeze()
        result_heatmap = torch.relu(result_heatmap) / torch.max(result_heatmap) # Normalisation
        result_heatmap = result_heatmap.numpy()
        result_heatmap = np.sum(result_heatmap, axis=0)
        result_heatmap = cv2.resize(result_heatmap, (720, 720))

        result_heatmap = result_heatmap * 3.0 # make heatmap more bright
        heatmap_colormap = cv2.applyColorMap(np.uint8(result_heatmap), cv2.COLORMAP_JET)
        alpha = 0.5  # Adjust the alpha value to control the intensity of the heatmap overlay
        combined_image = cv2.addWeighted(image, 1 - alpha, heatmap_colormap, alpha, 0)
        
        white_image = np.ones((720, 200, 3), dtype=np.uint8) * 255  
        # Concatenate the images horizontally
        final_image = np.hstack((image, white_image, combined_image))
        # Concatenate the images verticaly (to put the text)
        white_image = np.ones((200, 720+720+200, 3), dtype=np.uint8) * 255 
        final_image = np.vstack((final_image, white_image))
        text = f"Class: {class_name} | Confidence: {str(np.round(confidence_score * 100))}%"
        font_scale = 2.0
        thickness = 2 
        cv2.putText(final_image, text, (10, 800), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)

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