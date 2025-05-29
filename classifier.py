#import necessary classes/libraries
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image

#definition the neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 200)  #input layer (28x28 = 784 pixels)
        self.fc2 = nn.Linear(200, 10)   #output layer (10 classes)

    def forward(self, x):
        x = x.view(-1, 784)  #flatten the input
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model():
    #load the dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.MNIST(root='/home/b/blyken007/MLA1/MNIST_JPGS/trainingSet/trainingSet', 
                                               train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    #initialize the neural network
    net = Net()

    #definition of the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    #training loop
    print("Pytorch output...\n")
    for epoch in range(10):  #adjust the number of epochs as needed
        running_loss = 0.0
        total = 0
        correct = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
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

    print("Done!")

    #save the trained model
    torch.save(net.state_dict(), 'mnist_model.pth')

def load_and_predict(image_path):
    #load the trained model
    net = Net()
    net.load_state_dict(torch.load('mnist_model.pth'))
    net.eval()

    #load the  preprocessed image
    image = Image.open(image_path).convert('L')  #convert to grayscale
    image = transforms.ToTensor()(image)
    image = image.view(1, -1)  #flatten the image

    #predictions
    with torch.no_grad():
        outputs = net(image)
        _, predicted = torch.max(outputs.data, 1)

    return predicted.item()

if __name__ == "__main__":
    #train the model
    train_model()

    #testing I/O
    test_image_path = ''
    while(test_image_path != 'exit'):
        test_image_path = str(input("Please enter a filepath:\n"))
        if (test_image_path=='exit'):
            print("Exiting...")
            break
        predicted_digit = load_and_predict(test_image_path)
        print(f"Classifier: {predicted_digit}")