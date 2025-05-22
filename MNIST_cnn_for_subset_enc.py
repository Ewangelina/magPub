import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pickle

# Define transformations for the training and test sets
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])

# Load CIFAR-10 dataset
#train_dataset = pickle.load(open(dataset, 'rb'))
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the CNN model for feature extraction
class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.Conv2d(32, 64, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Linear(1600, 256)  # Change to desired feature size if needed (50 in this example)
        self.fc2 = nn.Linear(256, 16)
    
    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.shape[0], -1)  # Flatten the tensor
        x = self.fc(x)
        x = self.fc2(x)
        return x

cnn_model = CNNFeatureExtractor()

# Extract features using the CNN
def extract_features(model, loader):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            features.append(output)
            labels.append(target)
    return torch.cat(features), torch.cat(labels)

train_features, train_labels = extract_features(cnn_model, train_loader)

# Save extracted features and labels using pickle
with open('./data/extracted_train_16.sav', 'wb') as f:
    pickle.dump((train_features, train_labels), f)

train_features, train_labels = extract_features(cnn_model, test_loader)

# Save extracted features and labels using pickle
with open('./data/extracted_test_16.sav', 'wb') as f:
    pickle.dump((train_features, train_labels), f)
