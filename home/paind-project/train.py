import torch
from torch import nn
from torch import optim
from torchvision import transforms, models, datasets

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sb

from collections import OrderedDict

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                          std = [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(256),
                                     transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                          std = [0.229, 0.224, 0.225])])

test_transforms= transforms.Compose([transforms.Resize(256),
                                     transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                          std = [0.229, 0.224, 0.225])])


# TODO: Load the datasets with ImageFolder
train_dataset = datasets.ImageFolder(train_dir, transform = train_transforms)
valid_dataset = datasets.ImageFolder(valid_dir, transform = valid_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform = test_transforms) 

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = 64, shuffle = True)
validloader = torch.utils.data.DataLoader(valid_dataset, batch_size = 64, shuffle = True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size = 64, shuffle = True)

amount = input('How many hidden layers do you want?')
i = 0
list1 = []
for i in range(amount):
    hiddenlayers = input('Input your hidden layers.')
    list1 = list1.append(hiddenlayers)

import json

json1 = 'cat_to_name.json'

with open(json1, 'r') as f:
      cat_to_name = json.load(f)
import argparse

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--save_dir", action="store", dest="save_dir", default="." , help = "Set directory to save checkpoints")
    parser.add_argument("--model", action="store", dest="model", default="densenet121" , help = "Set architechture('densenet121' or 'vgg19')")
    parser.add_argument("--learning_rate", action="store", dest="lr", default=0.001 , help = "Set learning rate")
    parser.add_argument("--hidden_units", action="store", dest="hidden_units", default=512 , help = "Set number of hidden units")
    parser.add_argument("--epochs", action="store", dest="epochs", default=5 , help = "Set number of epochs")
    parser.add_argument("--gpu", action="store_true", dest="cuda", default=False , help = "Use CUDA for training")
    parser.add_argument('data_path', action="store")

    return parser.parse_args()

args = get_arguments()
model = args.model
epochs = args.epochs
learning_rate = args.lr
if args.arch == "densenet121":
    model = models.densenet121(pretrained=True)
elif args.arch == "vgg19":
    model = models.vgg19(pretrained=True)
else:
   ValueError('Architecture not supported')
for param in model.parameters():
    param.requires_grad = False
classifier = nn.Sequential(OrderedDict([
			  ('fc1', nn.Linear(1024, 500)),
			  ('relu', nn.reLU()),
			  ('fc2', nn.Linear(500, 2)),
			  ('output', nn.LogSoftmax(dim=1))
			  ]))
model.classifier = classifier
if args.cuda == True:
    device = 'cuda'
else: 
    device = 'cpu'
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)

def validation(model, testloader, criterion):
    test_loss = 0
    accuracy = 0
    for images, labels in testloader:                          

        images.resize_(images.shape[0], 1024)

        output = model.forward(images)
        test_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean() 

    return test_loss, accuracy
model.to(device)
running_loss = 0
print_every = 5 
steps = 0
for e in range(epochs):

    # puts it in training mode so that dropout is enabled
    model.train()                
                                                                             
    for images, labels in validloader:
        steps += 1
        
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        output = model.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            # Make sure network is in eval mode for inference and removes dropout
            model.eval()                                                                                     
            
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                test_loss, accuracy = validation(model, validloader, criterion)
                
            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Test Loss: {:.3f}.. ".format(test_loss/len(validloader)),
                  "Test Accuracy: {:.3f}".format(accuracy/len(validloader)))
            
            running_loss = 0
            
            # Make sure training is back on and dropout is enabled once more
            model.train()
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to("cuda"), labels.to("cuda")
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
checkpoint = {'state_dict' : model.state_dict(),
           'class_to_idx' : model.class_to_idx,
           'epoch' : args.epoch,
           'optimizer' : optimizer.state_dict,
           'Learning_Rate' : args.lr}
torch.save(checkpoint, 'checkpoint.pth')