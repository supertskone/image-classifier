import json

import torch
from torch import nn, optim
from torchvision import datasets, transforms, models


def train_model(data_dir, save_dir, category_names, arch='vgg16', hidden_units=512, epochs=5, lr=0.001):
    # Define data transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    }

    # Load the datasets with ImageFolder
    image_datasets = {
        phase: datasets.ImageFolder(root=data_dir + '/' + phase, transform=data_transforms[phase])
        for phase in ['train', 'valid', 'test']
    }

    # Using the image datasets and the transforms, define the dataloaders
    dataloaders = {
        phase: torch.utils.data.DataLoader(image_datasets[phase], batch_size=64, shuffle=True)
        for phase in ['train', 'valid', 'test']
    }

    # Load the mapping from category label to category name
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

    # Load the pre-trained model
    model = models.__dict__[arch](pretrained=True)

    # Freeze the pre-trained model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Replace the classifier with a new untrained feed-forward network
    input_size = model.classifier[0].in_features
    classifier = nn.Sequential(
        nn.Linear(input_size, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_units, len(cat_to_name)),
        nn.LogSoftmax(dim=1)
    )
    model.classifier = classifier

    # Define the loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

    # Move the model to the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Train the classifier layers using backpropagation
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        print(f"Epoch {epoch + 1}/{epochs}")
        for batch_idx, (inputs, labels) in enumerate(dataloaders['train'], 1):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            log_ps = model.forward(inputs)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(dataloaders['train'])}.. "
                      f"Train loss: {running_loss / batch_idx:.3f}")

        # Calculate validation loss and accuracy
        valid_loss = 0.0
        accuracy = 0.0
        model.eval()
        with torch.no_grad():
            for inputs, labels in dataloaders['valid']:
                inputs, labels = inputs.to(device), labels.to(device)
                log_ps = model.forward(inputs)
                valid_loss += criterion(log_ps, labels).item()
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        model.train()
        print(f"Epoch {epoch + 1}/{epochs}.. "
              f"Train loss: {running_loss / len(dataloaders['train']):.3f}.. "
              f"Validation loss: {valid_loss / len(dataloaders['valid']):.3f}.. "
              f"Validation accuracy: {accuracy / len(dataloaders['valid']):.3f}")

    # Save the checkpoint
    model.class_to_idx = image_datasets['train'].class_to_idx
    checkpoint = {
        'arch': arch,
        'classifier': classifier,
        'class_to_idx': model.class_to_idx,
        'state_dict': model.state_dict()
    }
    torch.save(checkpoint, save_dir + '/checkpoint.pth')


if __name__ == "__main__":
    data_dir = 'flower_data'
    save_dir = 'save_data'
    category_names = 'cat_to_name.json'
    train_model(data_dir, save_dir, category_names, hidden_units=512)