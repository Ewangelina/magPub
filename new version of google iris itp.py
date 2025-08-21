!pip install pykan
from google.colab import files
from google.colab import drive
drive.mount('/content/drive')

from sklearn import model_selection
import sklearn
from kan import *
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.datasets import load_breast_cancer, load_wine, load_iris, load_digits
from tensorflow.keras.datasets import mnist
from datetime import datetime
from torchvision import datasets, transforms
import pickle

def string_datetime():
    return str(datetime.now()).replace(":", "").replace("-", "").replace(".", "")

PATIENCE_VALUE = 5
TOLERANCE_AMOUNT = 0.00001
LOG_FILE_FILEPATH = '/content/drive/My Drive/text_logs/' + string_datetime() + '.txt'
#LOG_FILE_FILEPATH = './text_logs/' + string_datetime() + '.txt'
STEPS = 150
BATCH_SIZE = 100


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
dtype = torch.get_default_dtype()
result_type = torch.long

with open(LOG_FILE_FILEPATH, 'w') as f:
    line = string_datetime() + '\n'
    f.write(line)
    f.close()

def write_to_file(line):
    line = line + '\n'
    with open(LOG_FILE_FILEPATH, "a") as f:
        f.write(line)
        f.close()

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
  return (in_dim * out_dim) * (KAN_grid + KAN_degree + 3) + out_dim

def count_mlp_layer_parameters(in_dim, out_dim):
  return (in_dim * out_dim) + out_dim

def count_kan_parameters_full(KAN_width, KAN_degree, KAN_grid):
  kan_sum = 0
  for i in range(len(KAN_width) - 1):
    kan_sum += count_kan_layer_parameters(KAN_width[i], KAN_width[i+1], KAN_degree, KAN_grid)

  return kan_sum


def make_mlp(no_layers, layers_width):
  layers = []
  for i in range(no_layers):
    layers.append(layers_width)
  model = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=layers, activation='relu', solver='Adam', max_iter=1, early_stopping=False, batch_size=BATCH_SIZE)
  return model

def create_matching_mlp(KAN_width, KAN_degree, KAN_grid, plus=False):
  no_kan_parameters = count_kan_parameters_full(KAN_width, KAN_degree, KAN_grid)
  in_dim = KAN_width[0]
  out_dim = KAN_width[-1]
  layer_width = int(in_dim/2)
  normal_layer_depth = 2
  current_no_params = 0
  
  while current_no_params < no_kan_parameters:
    layer_width += 1
    current_no_params = count_mlp_layer_parameters(in_dim, layer_width) + count_mlp_layer_parameters(layer_width, out_dim) + normal_layer_depth * count_mlp_layer_parameters(layer_width, layer_width)
  
  if not plus:
    layer_width -= 1

  return make_mlp(normal_layer_depth, layer_width), normal_layer_depth, layer_width

def mlp_evaluate(mlp_model, x, y):
    mlp_predicted = mlp_model.predict(x)
    mlp_test_acc = 0
    for i in range(len(mlp_predicted)):
      if mlp_predicted[i] == y[i].item():
        total_correct += 1
    return total_correct / len(mlp_predicted)

def processDataset(data, not_random=None): #sklearn.datasets data
    y = data.target
    X = data.data
    X = sklearn.preprocessing.normalize(X)
    X_train_full, X_test, y_train_full, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.25, random_state=not_random)
    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=not_random)
    dataset = {}
    dataset['train_input'] = torch.from_numpy(np.array(X_train)).type(dtype).to(device)
    dataset['test_input'] = torch.from_numpy(np.array(X_val)).type(dtype).to(device)
    dataset['train_label'] = torch.from_numpy(np.array(y_train)).type(result_type).to(device)
    dataset['test_label'] = torch.from_numpy(np.array(y_val)).type(result_type).to(device)
    full_dataset = {}
    full_dataset['test_input'] = torch.from_numpy(np.array(X_test)).type(dtype).to(device)
    full_dataset['test_label'] = torch.from_numpy(np.array(y_test)).type(result_type).to(device)
    return dataset, full_dataset

def trainKAN(dataset, steps=1, KAN_width=None, KAN_grid=None, KAN_degree=3, KAN_lambda=0, KAN_sparse_init=False, last_best_acc=0, model=None):
    def train_acc():
        return torch.mean((torch.argmax(model(dataset['train_input']), dim=1) == dataset['train_label']).type(dtype))

    def test_acc():
        return torch.mean((torch.argmax(model(dataset['test_input']), dim=1) == dataset['test_label']).type(dtype))

    if model is None:
        model = KAN(width=KAN_width, grid=KAN_grid, k=KAN_degree, device=device, sparse_init=KAN_sparse_init)

    test_line = "Model recieved width: " + str(KAN_width)
    write_to_file(test_line)
    final_results = {}
    final_results['train_acc'] = []
    final_results['test_acc'] = []
    best_test_acc = last_best_acc
    patience = PATIENCE_VALUE
    for i in range(steps):
        fit_results = model.fit(dataset, opt='Adam', steps=1, lamb=KAN_lambda, metrics=(train_acc, test_acc), loss_fn=torch.nn.CrossEntropyLoss(), batch=BATCH_SIZE)
        final_results['train_acc'].append(fit_results['train_acc'][0])
        final_results['test_acc'].append(fit_results['test_acc'][0])
        if fit_results['test_acc'][0] < best_test_acc + TOLERANCE_AMOUNT:
            if patience == 0:
                return model, final_results, best_test_acc, i + 1
            else:
                patience -= 1
        else:
            patience = PATIENCE_VALUE
            best_test_acc = fit_results['test_acc'][0]
            print("best_test_acc: ", best_test_acc)

    return model, final_results, best_test_acc, steps

def trainMLP(dataset, model):
    best_test_acc = 0
    patience = PATIENCE_VALUE
    train_acc = 0
    val_acc = 0

    for i in range(STEPS):
        model = model.fit(dataset['train_input'], dataset['train_label'])
        train_acc = mlp_evaluate(model, dataset['train_input'], dataset['train_label'])
        val_acc = mlp_evaluate(model, dataset['test_input'], dataset['test_label'])


        line = "MLP - train_acc: " + str(train_acc) + " val_acc: " + str(val_acc)
        write_to_file(line)
        print("MLP - train_acc: ", train_acc, " val_acc: ", val_acc)

        if val_acc < best_test_acc + TOLERANCE_AMOUNT:
            if patience == 0:
                return model, train_acc, val_acc, i + 1
            else:
                patience -= 1
        else:
            patience = PATIENCE_VALUE
            best_test_acc = val_acc
            

    return model, train_acc, val_acc, STEPS


def write_model_parameters(model):
    line = ""
    act_fun_num = 1
    no_params_table = []
    for fun_part in model.act_fun:
        line += str(act_fun_num) + ' layer parameters:  '
        for el in fun_part.grid:
            line = line[:-1] + '\n'
            no_params = 0
            for elel in el:
                line += str(elel.item()) + ';'
                no_params += 1
            no_params_table.append(no_params)
        line = line[:-1] + '\n'
        act_fun_num += 1
    write_to_file(line[:-1])
    #If all layers have the same abount of grid elements
    if len(set(no_params_table)) == 1:
      return no_params
    else:
      write_to_file("Performed grid extension")
      write_to_file(str(no_params_table))
      return no_params

def write_model_formula(model):
    line = ""
    formulas = model.symbolic_formula()[0]
    for form in formulas:
        line += str(ex_round(form, 4)) + "\n"
    write_to_file(line[:-1])

skip = 0
# -----------------------------------

#dataset_name = "Mnist"
#dataset, full_dataset, in_dim, out_dim = processKerasDatasetMNIST()

data = load_iris()
dataset_name = "Iris"
write_to_file(str(data.target_names))
dataset, full_dataset = processDataset(data)
in_dim = len(data.feature_names)
out_dim = len(data.target_names)

write_to_file(dataset_name)

log_csv(["dataset_name", "kan_grid", "kan_grid_passed_parameter", "kan_degree", "kan_lam", "kan_width", "kan_params", "kan_time_difference", "kan_steps", "kan_train_acc", "kan_val_acc", "kan_test_acc", "mlp_test_acc", "mlp_val_acc", "mlp_train_acc", "mlp_steps", "mlp_time_difference", "mlp_params", "mlp_no_layers", "mlp_width_layers", "mlp2_test_acc", "mlp2_val_acc", "mlp2_train_acc", "mlp2_steps", "mlp2_time_difference", "mlp2_params", "no_layers2", "width_layers2"])

line = "Skipping " + str(skip) + " lines"
write_to_file(line)

KAN_width_table = []
KAN_grid_table = [3, 5, 9]
KAN_degree_table = [2, 3, 4, 5]
KAN_lambda_table = [0]

for i in range(int(in_dim * 0.75), (2*in_dim)+3, 2):
    current_width = [in_dim, i, out_dim]
    KAN_width_table.append(current_width)

for i in range(int(in_dim), (in_dim)+10, 2):
    for j in range(int(in_dim), (2*in_dim)+2, 2):
      current_width = [in_dim, i, j, out_dim]
      KAN_width_table.append(current_width)

#pre, post = "5 2 0 [13, 19, 27, 3] ".split("[")
#temp_width = [int(x) for x in post[:-2].split(", ")]
#temp_grid, temp_deg, temp_lamb = [int(x) for x in pre[:-1].split(" ")]

#KAN_width_table = [temp_width]
#KAN_grid_table = [temp_grid]
#KAN_degree_table = [temp_deg]
#KAN_lambda_table = [temp_lamb]


best_acc = 0
best_int_acc = 0
best_acc = False

for grid in KAN_grid_table:
    for degree in KAN_degree_table:
        for lam in KAN_lambda_table:
            for width in KAN_width_table:
              for i in range(5):
                if skip > 0:
                  skip -= 1
                  continue

                copied_width = width.copy()
                kan_start = string_datetime()
                intro_line = "------------- start ---- grid: " + str(grid) + " deg: " + str(degree) + " lambda: " + str(lam) + " width: " + str(copied_width) + " DT: " + kan_start
                write_to_file(intro_line)

                model, results, kan_val_acc, last_i = trainKAN(dataset, steps=STEPS, KAN_width=copied_width, KAN_grid=grid, KAN_degree=degree, KAN_lambda=lam, KAN_sparse_init=False)
                kan_test_acc = torch.mean((torch.argmax(model(full_dataset['test_input']), dim=1) == full_dataset['test_label']).type(dtype)).item()
                kan_train_acc = torch.mean((torch.argmax(model(dataset['train_input']), dim=1) == dataset['train_label']).type(dtype)).item()

                first_train_line = "normal train: " + str(last_i) + " steps; best val acc: " + str(kan_val_acc) + "; best TEST acc: " + str(kan_test_acc) + " train acc: " + str(kan_train_acc)
                write_to_file(first_train_line)

                accurate_grid_elements = write_model_parameters(model)

                mlp_start = kan_end = string_datetime()
                line = "MLP section start at " + kan_end
                write_to_file(line)

                mlp_model, no_layers, width_layers = create_matching_mlp(copied_width, degree, accurate_grid_elements, False):
                mlp_model, mlp_train_acc, mlp_val_acc, mlp_steps = trainMLP(dataset, mlp_model)
                mlp_test_acc = mlp_evaluate(mlp_model, full_dataset['test_input'], full_dataset['test_label'])
                
                mlp_start2 = mlp_end = string_datetime()
                line = "MLP MIN with " + str(no_layers) + " layers of width " + str(width_layers) + " achieved TEST accuracy of " + str(mlp_test_acc) + " val acc: " + str(mlp_val_acc) + " test acc: " + str(mlp_test_acc) + " at " + mlp_steps
                write_to_file(line)

                kan_params = count_kan_parameters_full(copied_width, degree, accurate_grid_elements)

                mlp_params = 0
                for l in range(no_layers):
                  mlp_params += count_mlp_layer_parameters(width_layers, width_layers)
                mlp_params += count_mlp_layer_parameters(width[0], width_layers) + count_mlp_layer_parameters(width_layers, width[-1])

                mlp2_model, no_layers2, width_layers2 = create_matching_mlp(copied_width, degree, accurate_grid_elements, False):
                mlp2_model, mlp2_train_acc, mlp2_val_acc, mlp2_steps = trainMLP(dataset, mlp2_model)
                mlp2_test_acc = mlp_evaluate(mlp2_model, full_dataset['test_input'], full_dataset['test_label'])
                
                mlp_end2 = string_datetime()
                line = "MLP MIN with " + str(no_layers2) + " layers of width " + str(width_layers2) + " achieved TEST accuracy of " + str(mlp2_test_acc) + " val acc: " + str(mlp2_val_acc) + " test acc: " + str(mlp2_test_acc) + " at " + mlp2_steps
                write_to_file(line)

                mlp2_params = 0
                for l in range(no_layers2):
                  mlp2_params += count_mlp_layer_parameters(width_layers2, width_layers2)
                mlp2_params += count_mlp_layer_parameters(width[0], width_layers2) + count_mlp_layer_parameters(width_layers2, width[-1])

                log_csv([dataset_name, str(accurate_grid_elements), str(grid), str(degree), str(lam), str(width), str(kan_params), time_difference(kan_start, kan_end), last_i, kan_train_acc, kan_val_acc, str(kan_test_acc), str(mlp_test_acc), mlp_val_acc, mlp_train_acc, mlp_steps, time_difference(mlp_start, mlp_end), str(mlp_params), str(no_layers), str(width_layers), str(mlp2_test_acc), mlp2_val_acc, mlp2_train_acc, mlp2_steps, time_difference(mlp_start2, mlp_end2), str(mlp2_params), str(no_layers2), str(width_layers2)])


print("I'm done accually!")


