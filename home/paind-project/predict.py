import torch
from torch import nn
from torch import optim
from torchvision import transforms, models, datasets

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sb

from collections import OrderedDict

json1 = 'cat_to_name.json'
cat_to_name = {}

with open(json1, 'r') as f:
      cat_to_name = json.load(f)
        
def get_arguments():
    parser = argparse.ArgumentParser(description='Recognize flowers')
    parser.add_argument('input', type=str, help='Image path to recognize')
    parser.add_argument('checkpoint', type=str, help='Model checkpoint path')
    parser.add_argument('--gpu', type=bool, default=False, help='Whether to use GPU or not')
    parser.add_argument('--class_names', type=str, default='cat_to_name.json', help='Path to class names mapping')
    parser.add_argument('--topk', type=int, default=5, help='Number of variants to show')

model = args.checkpoint
args = get_arguments()
if args.cuda == True:
    device = 'cuda'
else: 
    device = 'cpu'
if args.gpu == True:
    gpu = True
model.to(device)
if gpu and torch.cuda.is_available():
    model.cuda()
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    img_loader = transforms.Compose([transforms.Resize(256), 
                                    transforms.CenterCrop(224), 
                                    transforms.ToTensor()])
    
    pil_image = Image.open(image)
    pil_image = img_loader(pil_image).float()
    
    np_image = np.array(pil_image)    
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    np_image = np.transpose(np_image, (2, 0, 1))
            
    return np_image
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to(device)
    with torch.no_grad():
        output = model.forward(image.type(torch.FloatTensor).to(device))
    probabilities = torch.exp(output).cpu()  # used LogSoftmax so convert back
    top_probs, top_classes = probabilities.topk(topk)
    return top_probs.numpy()[0], [cat_to_name[str(cls)] for cls in top_classes.numpy()[0]]
images, labels = next(iter(args.input))
images, labels = images.to(device), labels.to(device)
prob, classes = predict(images, model, args.topk)
print(prob)
print(args.topk)
