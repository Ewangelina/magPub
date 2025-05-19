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
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Linear(576, 50)  # Change to desired feature size if needed (50 in this example)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
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

#train_features, train_labels = extract_features(cnn_model, train_loader)

with open('./data/extracted_train.sav', 'rb') as f:
    train_features, train_labels = pickle.load(f)

# Save extracted features and labels using pickle
with open('./data/extracted_train.sav', 'wb') as f:
    pickle.dump((train_features, train_labels), f)

train_features, train_labels = extract_features(cnn_model, test_loader)

# Save extracted features and labels using pickle
with open('./data/extracted_test.sav', 'wb') as f:
    pickle.dump((train_features, train_labels), f)
