import os

import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import json


def process_image(image_path):
    # Open the image using PIL
    img = Image.open(image_path)

    # Define transformations for the image
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Preprocess the image
    img = data_transforms(img)

    # Add a batch dimension to the image
    img = img.unsqueeze(0)

    return img


def load_checkpoint(filepath, hidden_units=512):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    checkpoint = torch.load(filepath, map_location='cpu')
    arch = checkpoint['arch']
    class_to_idx = checkpoint['class_to_idx']

    # Load the pre-trained model with the same architecture as the saved checkpoint
    model = models.__dict__[arch](pretrained=True)

    # Replace the classifier with a new untrained feed-forward network with the same architecture as during training
    input_size = model.classifier[0].in_features
    classifier = nn.Sequential(
        nn.Linear(input_size, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_units, len(class_to_idx)),
        nn.LogSoftmax(dim=1)
    )
    model.classifier = classifier

    # Load the state_dict into the model
    model.load_state_dict(checkpoint['state_dict'])

    # Move the model to the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, class_to_idx


def predict(image_path, model, class_to_idx, topk=5):
    # Set the model to evaluation mode
    model.eval()

    # Move the model to the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Preprocess the image
    img = process_image(image_path)

    # Move the input data to the GPU if available
    img = img.to(device)

    # Make predictions
    with torch.no_grad():
        output = model(img)
        ps = torch.exp(output)
        probs, indices = ps.topk(topk)
        probs = probs.squeeze().tolist()
        indices = indices.squeeze().tolist()

    # Convert indices to class labels
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    classes = [idx_to_class[idx] for idx in indices]

    return probs, classes
