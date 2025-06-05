import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pickle
import numpy as np

BATCH_SIZE = 1000
DATA_FILE_FILEPATH = "./data/"

def extract_features(loader):
    features = []
    labels = []
    with torch.no_grad():
        num = 0
        for data, target in loader:
            if target is None:
                break
            
            for i in range(BATCH_SIZE):
                tab_2d = data[i][0]
                inp = []
                for row in tab_2d:
                    for el in row:
                        inp.append(el)
                features.append(inp)
                labels.append(target[i])

            print("*", end="")

            x_filename = './data/X_' + str(num) + '.sav'
            y_filename = './data/y_' + str(num) + '.sav'
            num += 1

            with open(y_filename, 'wb') as f:
                pickle.dump(labels, f)
                labels = []
                f.close()

            
            with open(x_filename, 'wb') as f:
                pickle.dump(features, f)
                features = []
                f.close()


def processKerasDatasetMNIST(random_state=None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

    
    transform2 = transforms.Compose([
        transforms.ToTensor()
    ])

    #test_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform2)
    #test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    #X_test_full, y_test_full = extract_features(test_loader)

    train_loc = DATA_FILE_FILEPATH + "X_test_full.sav"
    dbfile = open(train_loc, 'rb')    
    X_test_full = pickle.load(dbfile)
    dbfile.close()

    el = X_test_full
    while True:
        print(len(el))
        el = el[0]

    in_dim = len(X_train_full[0])
    out_dim = 10

    X_train_full = np.array(X_train_full)
    y_train_full = np.array(y_train_full)
    X_test_full = np.array(X_test_full)
    y_test_full = np.array(y_test_full)

    normalized_train_X = X_train_full
    normalized_test_X = X_test_full

    X_train, X_val, y_train, y_val = model_selection.train_test_split(normalized_train_X, y_train_full, test_size=0.2, random_state=random_state)
    dataset = {}
    dataset['train_input'] = torch.from_numpy(np.array(X_train)).type(dtype).to(device)
    dataset['test_input'] = torch.from_numpy(np.array(X_val)).type(dtype).to(device)
    dataset['train_label'] = torch.from_numpy(np.array(y_train)).type(result_type).to(device)
    dataset['test_label'] = torch.from_numpy(np.array(y_val)).type(result_type).to(device)
    full_dataset = {}
    full_dataset['train_input'] = torch.from_numpy(np.array(normalized_train_X)).type(dtype).to(device)
    full_dataset['train_label'] = torch.from_numpy(np.array(y_train_full)).type(result_type).to(device)
    full_dataset['test_input'] = torch.from_numpy(np.array(normalized_test_X)).type(dtype).to(device)
    full_dataset['test_label'] = torch.from_numpy(np.array(y_test_full)).type(result_type).to(device)

    return dataset, full_dataset, in_dim, out_dim

processKerasDatasetMNIST()

if False:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

    train_dataset = datasets.MNIST(root=DATA_FILE_FILEPATH, train=True, download=True, transform=transform)
    X_train_full = []
    y_train_full = []
    for i in range(len(train_dataset)):
        this_picture = []
        for picture in train_dataset[i][0]:
            for row in picture:
                for cell in row:
                    this_picture.append(cell)

        X_train_full.append(this_picture)
        y_train_full.append(train_dataset[i][1])
        if i % 500 == 0:
          print('*', end='')

    X_train_full = np.array(X_train_full)
    y_train_full = np.array(y_train_full)
    
    normalized_train_X = X_train_full

    if True:
        filename = DATA_FILE_FILEPATH + "X_train_full.sav"
        with open(filename, 'wb') as f:
            pickle.dump(normalized_train_X, f)
            features = []
            f.close()

        filename = DATA_FILE_FILEPATH + "y_train_full.sav"
        with open(filename, 'wb') as f:
            pickle.dump(y_train_full, f)
            features = []
            f.close()


    print("new")

    test_dataset = datasets.MNIST(root=DATA_FILE_FILEPATH, train=False, download=True, transform=transform)
    X_test_full = []
    y_test_full = []
    for i in range(len(test_dataset)):
        this_picture = []
        for picture in test_dataset[i][0]:
            for row in picture:
                for cell in row:
                    this_picture.append(cell)
                
        X_test_full.append(this_picture)
        y_test_full.append(test_dataset[i][1])
        if i % 500 == 0:
          print('*', end='')


    X_test_full = np.array(X_test_full)
    y_test_full = np.array(y_test_full)

    normalized_test_X = X_test_full

    in_dim = len(X_train_full[0])
    out_dim = 10

    if True:
        filename = DATA_FILE_FILEPATH + "in_dim.sav"
        with open(filename, 'wb') as f:
            pickle.dump(in_dim, f)
            features = []
            f.close()

        filename = DATA_FILE_FILEPATH + "X_test_full.sav"
        with open(filename, 'wb') as f:
            pickle.dump(normalized_test_X, f)
            features = []
            f.close()
          
        filename = DATA_FILE_FILEPATH + "y_test_full.sav"
        with open(filename, 'wb') as f:
            pickle.dump(y_test_full, f)
            features = []
            f.close()
