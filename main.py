import torch
from torch import nn
from torchvision import utils
from torchvision.transforms import v2, ToTensor
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from fer2013_data_loading import fer2013LoadData, fer2013Dataset, ToTensor

get_data = fer2013LoadData("data/fer2013/fer2013.csv")
train_loader, private_loader, test_loader = get_data.read_data()

train_dataset = fer2013Dataset(train_loader, transform=ToTensor())
test_dataset = fer2013Dataset(test_loader, transform=ToTensor())

batch_size = 64


# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(48*48, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 7)
        )

    def forward(self, x):
        #x = self.flatten(x)
        x = x/256 # to lower elements and parameters
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.00001, momentum=0.9)
#optimizer = optim.SGD(model.parameters(), lr=0.7)


for epoch in range(20):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_dataset):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data['image'].to(device), data['emotion'].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        #print (outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item() #*inputs.size(0)
        if i % 20 == 19:    # print every 20 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 20:.3f}')
            #print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / len(train_loader)}')
            running_loss = 0.0

print('Finished Training')

# ### run forward pass - predicting

print ("Start predicting")

#inputsp = test_dataset.to(device)


with torch.no_grad():
    for i,data in enumerate(test_dataset):
        dev_data = data['image'].to(device)
        pred = model(dev_data)
        #image_data = np.reshape(np.array(test_loader[1][0].split(),dtype=int), (-1, 48))
        #plt.imshow(image_data)
        #plt.show()
        max_value = pred.max()
        print (test_loader[i][1], ((pred==max_value).nonzero(as_tuple=True)[0]))
        #print (test_loader[i][1], pred)



