# import necessary libraries 
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 200)  # input layer (28x28 = 784 pixels)
        self.fc2 = nn.Linear(200, 10)   # output layer (10 classes)

    def forward(self, x):
        x = x.view(-1, 784)  # flatten the input
        x = torch.relu(self.fc1(x))     # using ReLU
        x = self.fc2(x)
        return x

def train_model():
    # Data preprocessing
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load training data
    train_dataset = ImageFolder(
        root='/home/b/blyken007/MLA1/MNIST_JPGS/trainingSet/trainingSet',
        transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Initialize the neural network
    net = Net().to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    # Training loop
    print("Training started...\n")
    for epoch in range(10):  # Adjust the number of epochs as needed
        running_loss = 0.0
        total = 0
        correct = 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if i % 100 == 99:
                print(f"[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 100:.3f}, Accuracy: {100 * correct / total:.2f}%")
                running_loss = 0.0

    print("Training complete.")
    torch.save(net.state_dict(), 'mnist_model.pth')

def load_model():
    net = Net().to(device)
    net.load_state_dict(torch.load('mnist_model.pth', map_location=device))
    net.eval()
    return net

def load_and_predict(image_path, model):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    try:
        image = Image.open(image_path).convert('L')
    except FileNotFoundError:
        print("Error: File not found.")
        return None
    except Exception as e:
        print(f"Error opening image: {e}")
        return None

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)

    return predicted.item()

if __name__ == "__main__":
    # Train the model
    train_model()

    # Load the model once
    model = load_model()

    # Interactive prediction
    while True:
        test_image_path = input("Enter an image filepath or 'exit' to quit:\n")
        if test_image_path.strip().lower() == 'exit':
            print("Exiting...")
            break

        predicted_digit = load_and_predict(test_image_path, model)
        if predicted_digit is not None:
            print(f"Classifier Prediction: {predicted_digit}")
