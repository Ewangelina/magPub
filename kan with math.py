import sklearn
from kan import *
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

from sklearn import model_selection
from sklearn.datasets import load_breast_cancer, load_wine, load_iris, load_digits
from datetime import datetime

skip = 0
model = None
# -----------------------------------
data = load_iris()
dataset_name = "Iris"

KAN_width_table = []
KAN_grid_table = [2]
KAN_degree_table = [2]
KAN_lambda_table = [0]

def string_datetime():
    return str(datetime.now()).replace(":", "").replace("-", "").replace(".", "")

PATIENCE_VALUE = 100
TOLERANCE_AMOUNT = 0.01
#LOG_FILE_FILEPATH = '/content/drive/My Drive/text_logs/' + string_datetime() + "_" + dataset_name + '.txt'
LOG_FILE_FILEPATH = './text_logs/' + string_datetime() + '.txt'
STEPS = 10 #1000
BATCH_SIZE = -1
MLP_TEST_IN = 1
DELAY = 0

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
  if isinstance(in_dim, list):
    in_dim = in_dim[0]
  if isinstance(out_dim, list):
    in_dim = out_dim[0]

  return int((in_dim * out_dim) * (KAN_grid + KAN_degree + 3) + out_dim)

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
  if BATCH_SIZE == -1:
    return sklearn.neural_network.MLPClassifier(hidden_layer_sizes=layers, activation='relu', solver='adam', early_stopping=False)
  else:
    return sklearn.neural_network.MLPClassifier(hidden_layer_sizes=layers, activation='relu', solver='adam', early_stopping=False, batch_size=BATCH_SIZE)

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
    total_correct = 0
    for i in range(len(mlp_predicted)):
      if mlp_predicted[i] == y[i].item():
        total_correct += 1

    acc = float(total_correct / len(mlp_predicted))
    return acc

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
        model = KAN(width=KAN_width, grid=KAN_grid, k=KAN_degree, device=device, sparse_init=KAN_sparse_init, auto_save=False)

    final_results = {}
    final_results['train_acc'] = []
    final_results['test_acc'] = []
    best_test_acc = last_best_acc
    best_model = model
    patience = PATIENCE_VALUE

    for i in range(steps):
        fit_results = model.fit(dataset, opt='Adam', steps=1, lamb=KAN_lambda, metrics=(train_acc, test_acc), loss_fn=torch.nn.CrossEntropyLoss(), batch=BATCH_SIZE)
        final_results['train_acc'].append(fit_results['train_acc'][0])
        final_results['test_acc'].append(fit_results['test_acc'][0])
        if fit_results['test_acc'][0] < best_test_acc + TOLERANCE_AMOUNT:
            if patience == 0:
                return best_model, final_results, best_test_acc, i + 1
            else:
                patience -= 1
                if fit_results['train_acc'][0] == 1:
                  return best_model, final_results, best_test_acc, i + 1
        else:
            patience = PATIENCE_VALUE
            best_test_acc = fit_results['test_acc'][0]
            best_model = model

            if best_test_acc == 1:
              return best_model, final_results, best_test_acc, i + 1

            if fit_results['train_acc'][0] == 1:
              return best_model, final_results, best_test_acc, i + 1

    return best_model, final_results, best_test_acc, steps

def trainMLP(dataset, model):
    best_acc = 0
    best_model = model
    patience = PATIENCE_VALUE
    train_acc = 0
    val_acc = 0
    classes = np.unique(dataset['train_label'])
    delay = DELAY

    for i in range(STEPS):
        for y in range(MLP_TEST_IN):
          model = model.partial_fit(dataset['train_input'], dataset['train_label'], classes=classes)
        val_acc = mlp_evaluate(model, dataset['test_input'], dataset['test_label'])

        line = "MLP - val_acc: " + str(val_acc)
        write_to_file(line)

        if val_acc < best_acc + TOLERANCE_AMOUNT:
            if patience == 0:
                return best_model, best_acc, i + 1
            else:
                patience -= 1
        else:
            patience = PATIENCE_VALUE
            best_acc = val_acc
            best_model = model

            if best_acc == 1:
                return best_model, best_acc, i + 1

        if delay == 0:
            delay = DELAY
            mlp_train_acc = mlp_evaluate(mlp_model, dataset['train_input'], dataset['train_label'])
            line = "MLP - train_acc: " + str(mlp_train_acc)
            write_to_file(line)
            if mlp_train_acc == 1:
                return best_model, best_acc, i + 1
        else:
            delay -= 1

    return best_model, best_acc, STEPS


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

write_to_file(str(data.target_names))
dataset, full_dataset = processDataset(data)
in_dim = len(data.feature_names)
out_dim = len(data.target_names)

write_to_file(dataset_name)

log_csv(["dataset_name", "kan_grid", "kan_grid_passed_parameter", "kan_degree", "kan_lam", "kan_width", "kan_params", "kan_time_difference", "kan_steps", "kan_train_acc", "kan_val_acc", "kan_test_acc"])

line = "Skipping " + str(skip) + " lines"
write_to_file(line)

current_width = ([in_dim, int((in_dim - out_dim)/2), out_dim])
KAN_width_table.append(current_width)
#Ustawianie szerokości
if False:
    for i in range(int(in_dim * 0.75), (2*in_dim)+3, 5):
        current_width = [in_dim, i, out_dim]
        KAN_width_table.append(current_width)

    for i in range(int(in_dim), (in_dim)+10, 2):
        for j in range(int(in_dim), (2*in_dim)+2, 5):
          current_width = [in_dim, i, j, out_dim]
          KAN_width_table.append(current_width)



best_acc = 0
best_int_acc = 0
best_acc = False

for grid in KAN_grid_table:
    for degree in KAN_degree_table:
        for lam in KAN_lambda_table:
            for width in KAN_width_table:
              copied_width = width.copy()
              for i in range(1):
                if skip > 0:
                  skip -= 1
                  continue

                garbage_width = width.copy()
                kan_start = string_datetime()
                intro_line = "------------- start ---- grid: " + str(grid) + " deg: " + str(degree) + " lambda: " + str(lam) + " width: " + str(copied_width) + " DT: " + kan_start
                write_to_file(intro_line)

                model, results, kan_val_acc, last_i = trainKAN(dataset, steps=STEPS, KAN_width=garbage_width, KAN_grid=grid, KAN_degree=degree, KAN_lambda=lam, KAN_sparse_init=False)
                kan_test_acc = torch.mean((torch.argmax(model(full_dataset['test_input']), dim=1) == full_dataset['test_label']).type(dtype)).item()
                kan_train_acc = torch.mean((torch.argmax(model(dataset['train_input']), dim=1) == dataset['train_label']).type(dtype)).item()

                first_train_line = "normal train: " + str(last_i) + " steps; best val acc: " + str(kan_val_acc) + "; best TEST acc: " + str(kan_test_acc) + " train acc: " + str(kan_train_acc)
                write_to_file(first_train_line)
                write_to_file(str(results))

                accurate_grid_elements = write_model_parameters(model)
                kan_params = count_kan_parameters_full(copied_width, degree, accurate_grid_elements)
                #Dla MLP
                if False:
                    mlp_start = kan_end = string_datetime()
                    line = "MLP section start at " + kan_end
                    write_to_file(line)

                    mlp_model, no_layers, width_layers = create_matching_mlp(copied_width, degree, accurate_grid_elements, False)
                    mlp_model, mlp_val_acc, mlp_steps = trainMLP(dataset, mlp_model)
                    mlp_train_acc = mlp_evaluate(mlp_model, dataset['train_input'], dataset['train_label'])
                    mlp_test_acc = mlp_evaluate(mlp_model, full_dataset['test_input'], full_dataset['test_label'])

                    mlp_start2 = mlp_end = string_datetime()
                    line = "MLP MIN with " + str(no_layers) + " layers of width " + str(width_layers) + " achieved TEST accuracy of " + str(mlp_test_acc) + " val acc: " + str(mlp_val_acc) + " train acc: " + str(mlp_train_acc) + " at " + str(mlp_steps)
                    write_to_file(line)

                    

                    mlp_params = 0
                    for l in range(no_layers):
                      mlp_params += count_mlp_layer_parameters(width_layers, width_layers)
                    mlp_params += count_mlp_layer_parameters(copied_width[0], width_layers) + count_mlp_layer_parameters(width_layers, copied_width[-1])

                    mlp2_model, no_layers2, width_layers2 = create_matching_mlp(copied_width, degree, accurate_grid_elements, True)
                    mlp2_model, mlp2_val_acc, mlp2_steps = trainMLP(dataset, mlp2_model)
                    mlp2_train_acc = mlp_evaluate(mlp2_model, dataset['train_input'], dataset['train_label'])
                    mlp2_test_acc = mlp_evaluate(mlp2_model, full_dataset['test_input'], full_dataset['test_label'])

                    mlp_end2 = string_datetime()
                    line = "MLP MAX with " + str(no_layers2) + " layers of width " + str(width_layers2) + " achieved TEST accuracy of " + str(mlp2_test_acc) + " val acc: " + str(mlp2_val_acc) + " train acc: " + str(mlp2_train_acc) + " at " + str(mlp2_steps)
                    write_to_file(line)

                    mlp2_params = 0
                    for l in range(no_layers2):
                      mlp2_params += count_mlp_layer_parameters(width_layers2, width_layers2)
                    mlp2_params += count_mlp_layer_parameters(copied_width[0], width_layers2) + count_mlp_layer_parameters(width_layers2, copied_width[-1])

                log_csv([dataset_name, str(accurate_grid_elements), str(grid), str(degree), str(lam), str(copied_width), str(kan_params), time_difference(kan_start, kan_end), last_i, kan_train_acc, kan_val_acc, str(kan_test_acc)])


model.plot()
plt.show()

model.suggest_symbolic(0,0,0)

lib = ['x','x^2','x^3','x^4','exp','log','sqrt','tanh','sin','abs']
model.auto_symbolic(lib=lib)
line = ""
formulas = model.symbolic_formula()[0]
for form in formulas:
    line += str(ex_round(form, 4)) + "\n"
print(line[:-1])
