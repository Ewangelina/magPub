import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from functools import reduce
from torch.optim import Optimizer
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
from sklearn import model_selection
import sklearn
import numpy as np
from tensorflow.keras.datasets import mnist
from datetime import datetime
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from kan import *


skip = 0
# -----------------------------------
dataset_name = "Mnist"

KAN_width_table = []
KAN_grid_table = [3, 5, 9]
KAN_degree_table = [2, 3, 4, 5]
KAN_lambda_table = [0]

def string_datetime():
    return str(datetime.now()).replace(":", "").replace("-", "").replace(".", "")

DATA_FILE_FILEPATH = '/content/drive/My Drive/data'
#LOG_FILE_FILEPATH = '/content/drive/My Drive/text_logs/' + string_datetime() + "_" + dataset_name + '.txt'
LOG_FILE_FILEPATH = './text_logs/' + string_datetime() + '.txt'
PATIENCE_VALUE = 25
TOLERANCE_AMOUNT = 0.1
STEPS = 500
BATCH_SIZE = 1000
DELAY_EVALUATE = 0

KAN_CNN_MNIST_WIDTH_ORIGIN = 64
KAN_CNN_MNIST_WIDTH_END = 10

with open(LOG_FILE_FILEPATH, 'w') as f:
    line = string_datetime() + '\n'
    f.write(line)
    f.close()

def write_to_file(line):
    line = line + '\n'
    with open(LOG_FILE_FILEPATH, "a") as f:
        f.write(line)
        f.close()

write_to_file(f"This is KAN with {KAN_CNN_MNIST_WIDTH_ORIGIN} parameters (global pulling) and only one layer on local PC - batch size {BATCH_SIZE}")

class CNNKAN(nn.Module):
    def __init__(self, width1, grid1, k1, width2, grid2, k2):
        super(CNNKAN, self).__init__()
        width_copy = [KAN_CNN_MNIST_WIDTH_ORIGIN, width1[1], KAN_CNN_MNIST_WIDTH_END]
        print(f"Creating CNNKAN with width1={width1}, grid1={grid1}, k1={k1}, width2={width2}, grid2={grid2}, k2={k2}. Time is now {string_datetime()}")
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        
        #self.kan1 = KANLinear(64 * 8 * 8, 256)   #in features, out features, grid, k
        #self.kan2 = KANLinear(256, 10)
        self.kan1 = KAN(width=width_copy, grid=grid1, k=k1, device=device, sparse_init=False) #width=[3136, 256]
        ##self.kan1 = KAN(width=width1, grid=grid1, k=k1, device=device, sparse_init=False) #width=[3136, 256]
        ##self.kan2 = KAN(width=width2, grid=grid2, k=k2, device=device, sparse_init=False) #width=[256, 10]
        #self.kan1 = KANLinear(width1[0], width1[-1], grid1, k1)
        #self.kan2 = KANLinear(width2[0], width2[-1], grid2, k2)


    def forward(self, x):
        x = F.selu(self.conv1(x))
        x = self.pool1(x)
        x = F.selu(self.conv2(x))
        x = self.pool2(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.kan1(x)
        ##x = self.kan2(x)
        return x


class CNN(nn.Module):
    def __init__(self, in_params, middle_params, out_params):
        super(CNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(in_params, middle_params)
        self.fc2 = nn.Linear(middle_params, out_params)  # Final output layer

    def forward(self, x):
        # Convolutional layers
        x = F.selu(self.conv1(x))
        x = self.pool1(x)
        x = F.selu(self.conv2(x))
        x = self.pool2(x)
        
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))

        # Flattening the layer for the fully connected layer
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.selu(self.fc1(x))
        x = self.fc2(x)

        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
dtype = torch.get_default_dtype()
result_type = torch.long

def log_csv(values):
    csv_file = LOG_FILE_FILEPATH[:-4] + "_csv.txt"
    line = ""
    for el in values:
      line = line + str(el) + ";"
    line = line[0:-1] + "\n"
    with open(csv_file, "a") as f:
        f.write(line)
        f.close()

def microseconds_since_2025(date_string):
    #Turns curdate into the amount of miliseconds since 01.04.2025
    #20250415 042707608044
    #yyyymmdd hhmmss
    date_format = "%Y%m%d %H%M%S%f"
    current_datetime = datetime.strptime(date_string, date_format)
    #reference datetime
    reference_datetime = datetime(2025, 4, 1)
    difference = (current_datetime - reference_datetime).total_seconds() * 1_000_000
    return int(difference)

def time_difference(start, end):
    return str(microseconds_since_2025(end) - microseconds_since_2025(start))

def count_kan_layer_parameters(in_dim, out_dim, KAN_degree, KAN_grid):
  if isinstance(in_dim, list):
    in_dim = in_dim[0]
  if isinstance(out_dim, list):
    in_dim = out_dim[0]

  return int((in_dim * out_dim) * (KAN_grid + KAN_degree + 3) + out_dim)

def count_mlp_layer_parameters(in_params, middle_params, out_params):
  out = (in_params * middle_params) + middle_params
  out = (middle_params * out_params) + out_params
  return out

def count_kan_parameters_full(KAN_width, KAN_degree, KAN_grid):
  kan_sum = 0
  for i in range(len(KAN_width) - 1):
    kan_sum += count_kan_layer_parameters(KAN_width[i], KAN_width[i+1], KAN_degree, KAN_grid)

  return kan_sum

def make_mlp(in_dim, layer_width, out_dim):
  return (CNN(in_dim, layer_width, out_dim)).to(device)

def create_matching_mlp(KAN_width, KAN_degree, KAN_grid, plus=False):
  no_kan_parameters = count_kan_parameters_full(KAN_width, KAN_degree, KAN_grid)
  in_dim = KAN_width[0]
  out_dim = KAN_width[-1]
  layer_width = KAN_width[1]
  current_no_params = count_mlp_layer_parameters(in_dim, layer_width, out_dim)

  while current_no_params < no_kan_parameters:
    layer_width += 1
    current_no_params = count_mlp_layer_parameters(in_dim, layer_width, out_dim)

  if not plus:
    layer_width -= 1

  return make_mlp(in_dim, layer_width, out_dim), layer_width, no_kan_parameters

def mlp_evaluate(mlp_model, x, y):
    mlp_predicted = mlp_model.predict(x)
    total_correct = 0
    for i in range(len(mlp_predicted)):
      if mlp_predicted[i] == y[i].item():
        total_correct += 1
    return total_correct / len(mlp_predicted)

def train_one_step(model, train_loader, optimizer):
    model.train()
    i = 100
    for batch_idx, (data, target) in enumerate(train_loader):
        if i == 0:
            print(f"Batch start Time: {string_datetime()}")
            i = 100
        else:
            i -= 1
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()

def evaluate(model, test_data):
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.CrossEntropyLoss()(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct/len(test_loader.dataset)


    line = "Eval_results: Average loss: " + str(test_loss) + " Accuracy: " + str(accuracy)
    write_to_file(line)

    return accuracy

def train(train_data, test_data, model, steps=1):
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)

    patience = PATIENCE_VALUE
    evaluate_delay = DELAY_EVALUATE
    best_acc = 0
    best_model = model
    for i in range(steps):
        train_one_step(model, train_loader, optimizer)
        if evaluate_delay == 0:
          evaluate_delay = DELAY_EVALUATE
          print(f"Before evaluate time {string_datetime()}")
          accuracy = evaluate(model, test_data)
          if accuracy < best_acc + TOLERANCE_AMOUNT:
              if patience == 0:
                  return best_model, best_acc, i + 1
              else:
                  patience -= 1
          else:
              patience = PATIENCE_VALUE
              best_acc = accuracy
              best_model = model
        else:
          evaluate_delay -= 1

    return best_model, best_acc, steps

def processKerasDatasetMNIST(random_state=None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

    train_dataset = datasets.MNIST(root=DATA_FILE_FILEPATH, train=True, download=True, transform=transform) #[0][1] -> int, 6000 2 1 28 28
    test_dataset = datasets.MNIST(root=DATA_FILE_FILEPATH, train=False, download=True, transform=transform)
    no_samples = len(train_dataset)
    len_of_val = int(0.2 * no_samples)

    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [no_samples - len_of_val, len_of_val]) #generator=torch.Generator().manual_seed(1)

    return train_subset, val_subset, test_dataset


train_subset, val_subset, test_dataset = processKerasDatasetMNIST()

write_to_file(dataset_name)

log_csv(["dataset_name", "kan_grid 1", "kan_degree 1", "kan_lam 1", "kan_width 1", "kan_params", "kan_time_difference", "kan_steps", "kan_train_acc", "kan_val_acc", "kan_test_acc", "mlp_test_acc", "mlp_val_acc", "mlp_train_acc", "mlp_steps", "time_difference mlp1", "mlp_params", "width_layers", "mlp2_test_acc", "mlp2_val_acc", "mlp2_train_acc", "mlp2_steps", "mlp 2 time difference", "mlp2_params", "width_layers2"])
line = "Skipping " + str(skip) + " lines"
write_to_file(line)

if len(KAN_width_table) == 0:
    for i in range(KAN_CNN_MNIST_WIDTH_END, KAN_CNN_MNIST_WIDTH_ORIGIN, int(KAN_CNN_MNIST_WIDTH_ORIGIN/5)):
        KAN_width_table.append(i)
    print(KAN_width_table)


for grid in KAN_grid_table:
    for degree in KAN_degree_table:
        for lam in KAN_lambda_table:
            for width in KAN_width_table:
              for i in range(5):
                if skip > 0:
                  skip -= 1
                  continue

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                kan_start = string_datetime()
                intro_line = "------------- start ---- grid: " + str(grid) + " deg: " + str(degree) + " lambda: " + str(lam) + " width: " + str(width) + " DT: " + kan_start
                write_to_file(intro_line)

                width1 = [KAN_CNN_MNIST_WIDTH_ORIGIN, width]
                width2 = [width, KAN_CNN_MNIST_WIDTH_END]
                width_copy = [KAN_CNN_MNIST_WIDTH_ORIGIN, width, KAN_CNN_MNIST_WIDTH_END]

                model = CNNKAN(width1, grid, degree, width2, grid, degree).to(device)
                best_model, kan_val_acc, steps = train(train_subset, val_subset, model, STEPS)
                kan_train_acc = evaluate(best_model, train_subset)
                kan_test_acc = evaluate(best_model, test_dataset)
                mlp_start = kan_end = string_datetime()

                first_train_line = "normal train: " + str(steps) + " steps; best val acc: " + str(kan_val_acc) + "; best TEST acc: " + str(kan_test_acc) + "; train acc: " + str(kan_train_acc) + " at " + kan_end
                write_to_file(first_train_line)

                mlp_model, width_layers, kan_params = create_matching_mlp(width_copy, degree, grid, False)
                mlp_model, mlp_val_acc, mlp_steps = train(train_subset, val_subset, mlp_model, STEPS)
                mlp_train_acc = evaluate(mlp_model, train_subset)
                mlp_test_acc = evaluate(mlp_model, test_dataset)

                mlp_start2 = mlp_end = string_datetime()
                line = "MLP MIN of width " + str(width_layers) + " achieved TEST accuracy of " + str(mlp_test_acc) + " val acc: " + str(mlp_val_acc) + " train acc: " + str(mlp_train_acc) + " at " + str(mlp_steps)
                write_to_file(line)

                mlp_params = count_mlp_layer_parameters(KAN_CNN_MNIST_WIDTH_ORIGIN, width_layers, KAN_CNN_MNIST_WIDTH_END)

                mlp2_model, width_layers2, kan_parameters = create_matching_mlp(width_copy, degree, grid, True)
                mlp2_model, mlp2_val_acc, mlp2_steps = train(train_subset, val_subset, mlp2_model, STEPS)
                mlp2_train_acc = evaluate(mlp2_model, train_subset)
                mlp2_test_acc = evaluate(mlp2_model, test_dataset)

                mlp_end2 = string_datetime()
                line = "MLP MAX of width " + str(width_layers2) + " achieved TEST accuracy of " + str(mlp2_test_acc) + " val acc: " + str(mlp2_val_acc) + " train acc: " + str(mlp2_train_acc) + " at " + str(mlp2_steps)
                write_to_file(line)

                mlp2_params = count_mlp_layer_parameters(KAN_CNN_MNIST_WIDTH_ORIGIN, width_layers2, KAN_CNN_MNIST_WIDTH_END)

                log_csv([dataset_name, grid, degree, lam, width_copy, kan_params, time_difference(kan_start, kan_end), steps, kan_train_acc, kan_val_acc, kan_test_acc, mlp_test_acc, mlp_val_acc, mlp_train_acc, mlp_steps, time_difference(mlp_start, mlp_end), mlp_params, width_layers, mlp2_test_acc, mlp2_val_acc, mlp2_train_acc, mlp2_steps, time_difference(mlp_start2, mlp_end2), mlp2_params, width_layers2])

print("I'm done accually!")


