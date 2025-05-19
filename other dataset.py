import sklearn
from kan import *
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.datasets import load_breast_cancer, load_wine, load_iris, load_digits, load_diabetes
from datetime import datetime
from sklearn.neural_network import MLPRegressor

def string_datetime():
    return str(datetime.now()).replace(":", "").replace("-", "").replace(".", "")

PATIENCE_VALUE = 7
TOLERANCE_AMOUNT = 0.0001
LOG_FILE_FILEPATH = './text_logs/' + string_datetime() + '.txt'
STEPS = 250


#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') !!!!!!!!!!!!!!!!!!!!
device = torch.device('cpu')
dtype = torch.get_default_dtype()
result_type = dtype

with open(LOG_FILE_FILEPATH, 'w') as f:
    line = string_datetime() + '\n'
    f.write(line)
    f.close()

def write_to_file(line):
    line = line + '\n'
    with open(LOG_FILE_FILEPATH, "a") as f:
        f.write(line)
        f.close()

def log_csv(dataset_name, kan_grid, kan_degree, kan_lamb, kan_width, kan_params, kan_time, kan_test_acc, mlp_test_acc, mlp_time, mlp_params, mlp_layers, mlp_width, mlp_test_acc2, mlp2_time_difference, mlp_params2, no_layers2, width_layers2):
    csv_file = LOG_FILE_FILEPATH[:-4] + "_csv.txt"
    line = dataset_name + ";" + kan_grid + ";" + kan_degree + ";" + kan_lamb + ";" + kan_width + ";"  + kan_params + ";" + kan_time + ";" + kan_test_acc + ";" + mlp_test_acc + ";" + mlp_time + ";" + mlp_params + ";" + mlp_layers + ";" + mlp_width + ";" + mlp_test_acc2 + ";" + mlp2_time_difference + ";" + mlp_params2 + ";" + no_layers2 + ";" + width_layers2 + '\n'
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

def make_mlp_min_layers(no_layers, layers_width):
  layers = []
  for i in range(no_layers):
    layers.append(layers_width)
  model = MLPRegressor(hidden_layer_sizes=layers, activation='relu', solver='lbfgs', max_iter=STEPS, tol=TOLERANCE_AMOUNT, early_stopping=True, validation_fraction=0.15, n_iter_no_change=PATIENCE_VALUE)
  return model

def decide_min_mlp_layers_and_make(in_dim, out_dim, no_kan_parameters, how_many_layers_is_min=2):
  layer_width = int(in_dim/2)
  normal_layer_depth = 0

  for i in range(1,how_many_layers_is_min+1):
    normal_layer_depth = i

    current_no_params = count_mlp_layer_parameters(in_dim, layer_width) + count_mlp_layer_parameters(layer_width, out_dim) + normal_layer_depth * count_mlp_layer_parameters(layer_width, layer_width)
    if current_no_params >= no_kan_parameters:
      return make_mlp_min_layers(normal_layer_depth, layer_width), normal_layer_depth, layer_width, True

  while current_no_params < no_kan_parameters:
    layer_width += 1
    current_no_params = count_mlp_layer_parameters(in_dim, layer_width) + count_mlp_layer_parameters(layer_width, out_dim) + normal_layer_depth * count_mlp_layer_parameters(layer_width, layer_width)

  return make_mlp_min_layers(normal_layer_depth, layer_width-1), normal_layer_depth, layer_width-1, False

def create_matching_mlp_min_layers(KAN_width, KAN_degree, KAN_grid, how_many_layers_is_min=2):
  kan_sum = 0
  for i in range(len(KAN_width) - 1):
    kan_sum += count_kan_layer_parameters(KAN_width[i], KAN_width[i+1], KAN_degree, KAN_grid)

  return decide_min_mlp_layers_and_make(KAN_width[0], KAN_width[-1], kan_sum, how_many_layers_is_min)

def decide_min_plus_mlp_layers_and_make(in_dim, out_dim, no_kan_parameters, how_many_layers_is_min=2):
  layer_width = int(in_dim/2)
  normal_layer_depth = 0

  for i in range(1,how_many_layers_is_min+1):
    normal_layer_depth = i

    current_no_params = count_mlp_layer_parameters(in_dim, layer_width) + count_mlp_layer_parameters(layer_width, out_dim) + normal_layer_depth * count_mlp_layer_parameters(layer_width, layer_width)
    if current_no_params >= no_kan_parameters:
      return make_mlp_min_layers(normal_layer_depth, layer_width), normal_layer_depth, layer_width, True

  while current_no_params < no_kan_parameters:
    layer_width += 1
    current_no_params = count_mlp_layer_parameters(in_dim, layer_width) + count_mlp_layer_parameters(layer_width, out_dim) + normal_layer_depth * count_mlp_layer_parameters(layer_width, layer_width)

  return make_mlp_min_layers(normal_layer_depth, layer_width), normal_layer_depth, layer_width, False

def create_matching_mlp_min_plus_layers(KAN_width, KAN_degree, KAN_grid, how_many_layers_is_min=2):
  kan_sum = 0
  for i in range(len(KAN_width) - 1):
    kan_sum += count_kan_layer_parameters(KAN_width[i], KAN_width[i+1], KAN_degree, KAN_grid)

  return decide_min_plus_mlp_layers_and_make(KAN_width[0], KAN_width[-1], kan_sum, how_many_layers_is_min)

def make_mlp_model_max_layers(in_dim, no_layers):
  return make_mlp_min_layers(no_layers, in_dim)

def create_matching_mlp_max_layers(KAN_width, KAN_degree, KAN_grid):
  kan_sum = 0
  for i in range(len(KAN_width) - 1):
    kan_sum += count_kan_layer_parameters(KAN_width[i], KAN_width[i+1], KAN_degree, KAN_grid)

  usual_mlp_layer = count_mlp_layer_parameters(KAN_width[0], KAN_width[0])
  no_layers = 2
  while usual_mlp_layer * no_layers < kan_sum:
    no_layers += 1
  no_layers -= 1
  return make_mlp_model_max_layers(KAN_width[0], no_layers), no_layers

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
    full_dataset['train_input'] = torch.from_numpy(np.array(X_train_full)).type(dtype).to(device)
    full_dataset['train_label'] = torch.from_numpy(np.array(y_train_full)).type(result_type).to(device)
    full_dataset['test_input'] = torch.from_numpy(np.array(X_test)).type(dtype).to(device)
    full_dataset['test_label'] = torch.from_numpy(np.array(y_test)).type(result_type).to(device)
    return dataset, full_dataset

def calculate_MSE(data, answers, predict_function):
    predicted = predict_function(data)
    suma = 0

    for i in range(len(predicted)):
        if torch.is_tensor(predicted[i]):
            predicted[i] = predicted[i].item()
        suma += (predicted[i] - answers[i])**2

    result = suma/len(predicted)
    return result

def trainKAN(dataset, steps=1, KAN_width=None, KAN_grid=None, KAN_degree=3, KAN_lambda=0, KAN_sparse_init=False, last_best_acc=99999999999999999, model=None):
    def train_acc():
        return calculate_MSE(dataset['train_input'], dataset['train_label'], model)

    def test_acc():
        return calculate_MSE(dataset['test_input'], dataset['test_label'], model)

    if model is None:
        model = KAN(width=KAN_width, grid=KAN_grid, k=KAN_degree, device=device, sparse_init=KAN_sparse_init)

    write_to_file(test_line)
    final_results = {}
    final_results['train_acc'] = []
    final_results['test_acc'] = []
    best_test_acc = last_best_acc
    patience = PATIENCE_VALUE
    best_model = None
    for i in range(steps):
        fit_results = model.fit(dataset, opt='LBFGS', steps=1, lamb=KAN_lambda, metrics=(train_acc, test_acc))
        final_results['train_acc'].append(fit_results['train_acc'][0])
        final_results['test_acc'].append(fit_results['test_acc'][0])
        if fit_results['test_acc'][0] > best_test_acc - TOLERANCE_AMOUNT:
            if patience == 0:
                return best_model, final_results, best_test_acc, i + 1
            else:
                patience -= 1
        else:
            patience = PATIENCE_VALUE
            best_test_acc = fit_results['test_acc'][0]
            best_model = model
            print("best_test_acc: ", best_test_acc)

    return model, final_results, best_test_acc, steps

def trainMLP(dataset, model):
  model = model.fit(dataset['train_input'], dataset['train_label'])
  return model


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

data = load_diabetes()
dataset_name = "Diabetes"
#write_to_file(str(data.target_names))
write_to_file(dataset_name)

dataset, full_dataset = processDataset(data)
in_dim = len(data.feature_names)
out_dim = 1

log_csv("dataset_name", "kan_grid", "kan_degree", "kan_lam", "kan_width", "kan_params", "kan_time_difference", "kan_test_MSE", "mlp_test_MSE", "mlp_time_difference", "mlp_params", "mlp_no_layers", "mlp_width_layers", "mlp_test_MSE2", "mlp2_time_difference", "mlp_params2", "no_layers2", "width_layers2")



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
                model, results, best_val_acc, last_i = trainKAN(dataset, steps=STEPS, KAN_width=copied_width, KAN_grid=grid, KAN_degree=degree, KAN_lambda=lam, KAN_sparse_init=False)
                kan_test_acc = calculate_MSE(full_dataset['test_input'], full_dataset['test_label'], model).item()
                first_train_line = "normal train: " + str(last_i) + " steps; final TEST MSE: " + str(kan_test_acc)
                write_to_file(first_train_line)
                accurate_grid_elements = write_model_parameters(model)
                if best_acc < best_val_acc:
                    write_to_file("New best ACC achieved")
                    best_acc = best_val_acc
                    if best_acc == 1:
                      write_to_file("BEST KAN ACC ACHIEVED")


                mlp_start = kan_end = string_datetime()
                line = "MLP section start at " + kan_end
                write_to_file(line)

                min_layers = 2
                mlp_model, no_layers, width_layers, surpassedKan = create_matching_mlp_min_layers(width, degree, accurate_grid_elements, min_layers)
                mlp_model = trainMLP(full_dataset, mlp_model)
 
                mlp_test_acc = calculate_MSE(full_dataset['test_input'], full_dataset['test_label'], mlp_model.predict).item()
                mlp_start2 = mlp_end = string_datetime()
                line = "MLP with " + str(no_layers) + " layers of width " + str(width_layers) + "(MIN); achieved TEST MSE of " + str(mlp_test_acc) + " at " + mlp_end
                write_to_file(line)

                kan_params = 0
                for l in range(len(width)-1):
                  kan_params += count_kan_layer_parameters(width[l], width[l+1], degree, grid)

                mlp_params = 0
                for l in range(no_layers):
                  mlp_params += count_mlp_layer_parameters(width_layers, width_layers)
                mlp_params += count_mlp_layer_parameters(width[0], width_layers) + count_mlp_layer_parameters(width_layers, width[-1])



                mlp_model2, no_layers2, width_layers2, surpassedKan2 = create_matching_mlp_min_plus_layers(width, degree, accurate_grid_elements, min_layers)
                mlp_model2 = trainMLP(full_dataset, mlp_model2)
  
                mlp_test_acc2 = calculate_MSE(full_dataset['test_input'], full_dataset['test_label'], mlp_model2.predict).item()
                mlp_end2 = string_datetime()
                line2 = "MLP with " + str(no_layers2) + " layers of width " + str(width_layers2) + "(M +); achieved TEST MSE of " + str(mlp_test_acc2) + " at " + mlp_end2
                write_to_file(line2)

                kan_params2 = 0
                for l in range(len(width)-1):
                  kan_params2 += count_kan_layer_parameters(width[l], width[l+1], degree, grid)

                mlp_params2 = 0
                for l in range(no_layers2):
                  mlp_params2 += count_mlp_layer_parameters(width_layers2, width_layers2)
                mlp_params2 += count_mlp_layer_parameters(width[0], width_layers2) + count_mlp_layer_parameters(width_layers2, width[-1])

                log_csv(dataset_name, str(grid), str(degree), str(lam), str(width), str(kan_params), time_difference(kan_start, kan_end), str(kan_test_acc), str(mlp_test_acc), time_difference(mlp_start, mlp_end), str(mlp_params), str(no_layers), str(width_layers), str(mlp_test_acc2), time_difference(mlp_start2, mlp_end2), str(mlp_params2), str(no_layers2), str(width_layers2))

                exit(0)







