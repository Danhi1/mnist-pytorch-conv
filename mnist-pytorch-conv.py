import torch 
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Loading data
train = datasets.MNIST("", train = True, download = True, 
                       transform = transforms.Compose([transforms.ToTensor()])) 
test = datasets.MNIST("", train = False, download = True, 
                       transform = transforms.Compose([transforms.ToTensor()])) 

train_set = torch.utils.data.DataLoader(train, batch_size = 50, shuffle = True)
test_set = torch.utils.data.DataLoader(test, batch_size = 50, shuffle = True)


# Defining our model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, 5) 
        self.conv2 = nn.Conv2d(16, 32, 5)
        
        # Flattening is neccessary between a conv and a linear layer
        self.flatten = nn.Flatten()
        
        # Fully connected (linear) layers
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 10)
        # Last output layer should have 10 classes (Numbers 0-9)
        
    # Forward function is used during training / testing and defines how the data flows through the nn
    def forward(self, x):
        # Using max pooling after applying ReLu normalization to our input, twice
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        
        # Flattening the data
        x = self.flatten(x)
        
        # Linear layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        # Activation function
        return F.log_softmax(x, dim = 1)
    
# Creating a network object    
net = Net()
# Specifying the optimizer
optimizer = optim.Adam(net.parameters(), lr = 0.003)
# Number of EPOCHS to train model for
EPOCHS = 8

# Makes sure the model is in the training mode
net.train()    

for epoch in range(EPOCHS):
    for data in train_set:
        # Extracts image and correct label from the data
        X, y = data
        # It's neccesarry to zero the gradient during training
        net.zero_grad()
        # Pass the data through the nn
        output = net(X)
        # Backpropagate the loss
        loss = F.nll_loss(output, y)
        loss.backward()
        optimizer.step()
    print(loss)

print("          TRAINING FINISHED          ")


# Testing the nn
correct = 0
total = 0
# Switch the nn into evaluation mode
net.eval()
# Disable grads to save time
with torch.no_grad():
    for data in test_set:
        X, y = data
        output = net(X)
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct+=1
            total+=1

print("Accuracy: ", round(correct/total, 3))
        
        
