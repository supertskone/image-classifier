# Image Classifier

The aim of the project was to build an image classifier on the 102 Category Flower Dataset, and then predict new flower images using the trained model.

If you do not find the flowers/ dataset in the current directory, /workspace/home/aipnd-project/, you can download it using the following commands.

!wget 'https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz'
!unlink flowers
!mkdir flowers && tar -xzf flower_data.tar.gz -C flowers

#Data Preparation & Label Mapping
For ensuring proper training of my neural network, I began by organizing the training images into folders labeled with numerical names representing their classes. These folders were further categorized into training, testing, and validation sets as follows:

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

It is important to note that the folder names do not directly correspond to the actual flower names but rather to numerical labels. To address this, Udacity provided a mapping file, cat_to_name.json, which associates the folder labels (1-102) with the respective flower names.

Next, I adapted the images to be compatible with the pre-trained networks from the torchvision library, which were originally trained on the ImageNet dataset. To achieve this, I defined transformations for the image data, including resizing them to 224x224 pixels. Subsequently, I created Torch Dataset objects using ImageFolder as follows:

# Loading the datasets using ImageFolder
train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)
valid_data = datasets.ImageFolder(data_dir + '/valid', transform=test_transforms)

Finally, I established Data Loader objects to efficiently work with the data.

Pre-trained Model Upload & Classifier Preparation
The fully-connected layer used for training on the flower images was as follows:

classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('dropout1', nn.Dropout(0.05)),
                          ('fc2', nn.Linear(hidden_units, no_output_categories)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

For this project, I selected the highly accurate VGG16 Convolutional Network, although its training time was considerably slow (approximately 30 minutes per epoch). Notably, I had previously defined the hidden_units as 4096, while no_output_categories corresponded to the length of cat_to_name.json, which is 102.

# Training & Testing the Network
To train the network, I set the hyperparameters, including the number of epochs, learning rate, etc. Opting for 10 epochs helped avoid overfitting. The code iterated through each epoch, training 20 batches at a time (each batch containing 64 images), and assessing the model's progress on the validation data. Finally, the training and validation metrics were printed.

## train:  
python app.py flower_data save_data --arch vgg16 --hidden_units 512 --epochs 5 --lr 0.001
## predict: 
python app.py flower_data save_data/checkpoint.pth --image_path flower_data/test/1/image_06743.jpg --topk 5

To evaluate the model's performance, I utilized a separate test dataset that the model had not encountered during training. The results were highly satisfactory, surpassing the target accuracy of approximately 70%.
