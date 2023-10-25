import torchvision.transforms as transforms
import torch
from torchvision import datasets, transforms
import torch.nn as nn
from lucent.modelzoo import inceptionv1
import copy


# Define the number of classes for new task
num_classes = 10
train_data_path = 'Data\Train_data'
test_data_path = 'Data\Test_data'

# Preprocessing of the pictures
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # Was taken from imagenet
])

# Load the traint dataset
dataset = datasets.ImageFolder(root=train_data_path, transform=preprocess)

# Load the evaluation dataset
eval_dataset = datasets.ImageFolder(root=test_data_path, transform=preprocess)
eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_accuracy = 0
print(device)
first_run = True
# hyperparameters research
for lr in [0.1, 0.01, 0.001]:
    for batch_size in [5, 10, 20, 40, 80]:
        print(lr, batch_size)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Define model
        model = inceptionv1(pretrained=True)
        # Create a new fully connected layer
        new_fc = nn.Linear(in_features=model.softmax2_pre_activation_matmul.in_features, out_features=num_classes)
        # Replace the last layer of the pre-trained model with the new layer
        model.softmax2_pre_activation_matmul = new_fc
        # Set the model to training mode
        model.train()
        # Freeze all layers except the last layer
        for name, param in model.named_parameters():
            if name not in ["softmax2_pre_activation_matmul.weight", "softmax2_pre_activation_matmul.bias"]:
                param.requires_grad = False

        # Define an optimizer and a loss function
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        criterion = torch.nn.CrossEntropyLoss()

        # Train for a certain number of epochs
        num_epochs = 20  # Adjust as needed
        best_loss = 100
        model = model.to(device)
        best_with_this_hyperparams = copy.deepcopy(model)
        if first_run:
            best_model = copy.deepcopy(model)
            first_run = False
        for epoch in range(num_epochs):
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            if(best_loss > loss.item() or best_loss == 100):
                best_loss = loss.item()
                best_with_this_hyperparams = copy.deepcopy(model)
            print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {loss.item()}')

        # Set the model to evaluation mode
        best_with_this_hyperparams.eval()
        correct = 0
        total = 0
        # Disable gradient computation for evaluation
        with torch.no_grad():
            for inputs, labels in eval_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = best_with_this_hyperparams(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Calculate accuracy
        accuracy = 100 * correct / total
        print(f'Accuracy on the evaluation dataset: {accuracy:.2f}%')
        if(accuracy > best_accuracy):
            best_model = copy.deepcopy(best_with_this_hyperparams)
            best_accuracy = accuracy


best_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in eval_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = best_model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Calculate accuracy
accuracy = 100 * correct / total
print(f'Best accuracy: {accuracy:.2f}%')
torch.save(best_model.state_dict(), 'model/pretrained_model_weights.pth')