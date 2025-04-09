import sklearn
from kan import *
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.datasets import load_breast_cancer, load_wine, load_iris
import datetime

def string_datetime():
    return str(datetime.datetime.now()).replace(":", "").replace("-", "").replace(".", "")

PATIENCE_VALUE = 3
TOLERANCE_AMOUNT = 0
LOG_FILE_FILEPATH = '.\\text_logs\\' + string_datetime() + '.txt'
STEPS = 30


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.get_default_dtype()
result_type = torch.long

def write_to_file(line):
    line = line + '\n'
    with open(LOG_FILE_FILEPATH, "a") as f:
        f.write(line)
        f.close()

def processDataset(data, not_random=None): #sklearn.datasets data
    y = data.target
    X = data.data
    X = sklearn.preprocessing.normalize(X)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.25, random_state=not_random) #random_state=7
    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X_train, y_train, test_size=0.33, random_state=not_random)
    dataset = {}
    dataset['train_input'] = torch.from_numpy(np.array(X_train)).type(dtype).to(device)
    dataset['test_input'] = torch.from_numpy(np.array(X_val)).type(dtype).to(device)
    dataset['train_label'] = torch.from_numpy(np.array(y_train)).type(result_type).to(device)
    dataset['test_label'] = torch.from_numpy(np.array(y_val)).type(result_type).to(device)
    test = {}
    test['test_input'] = torch.from_numpy(np.array(X_test)).type(dtype).to(device)
    test['test_label'] = torch.from_numpy(np.array(y_test)).type(result_type).to(device)
    return dataset, test

def trainKAN(dataset, steps=1, KAN_width=None, KAN_grid=None, KAN_degree=3, KAN_lambda=0, KAN_sparse_init=False, last_best_acc=0, model=None):
    def train_acc():
        return torch.mean((torch.argmax(model(dataset['train_input']), dim=1) == dataset['train_label']).type(dtype))

    def test_acc():
        return torch.mean((torch.argmax(model(dataset['test_input']), dim=1) == dataset['test_label']).type(dtype))

    if model is None:
        model = KAN(width=KAN_width, grid=KAN_grid, k=KAN_degree, device=device, sparse_init=KAN_sparse_init)

    test_line "Model recieved width: " + str(KAN_width)
    write_to_file(test_line)
    final_results = {}
    final_results['train_acc'] = []
    final_results['test_acc'] = []
    best_test_acc = last_best_acc
    patience = PATIENCE_VALUE
    for i in range(steps):
        fit_results = model.fit(dataset, opt='LBFGS', steps=1, lamb=KAN_lambda, metrics=(train_acc, test_acc), loss_fn=torch.nn.CrossEntropyLoss())
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

def write_model_parameters(model):
    line = ""
    act_fun_num = 1
    for fun_part in model.act_fun:
        line += str(act_fun_num) + ' layer parameters:  '
        for el in fun_part.grid:
            line = line[:-1] + '\n'
            for elel in el:
                line += str(elel.item()) + ';'
        line = line[:-1] + '\n'
        act_fun_num += 1
    write_to_file(line[:-1])

def write_model_formula(model):
    line = ""
    formulas = model.symbolic_formula()[0]
    for form in formulas:
        line += str(ex_round(form, 4)) + "\n"
    write_to_file(line[:-1])


    


data = load_iris()
dataset, test = processDataset(data)
in_dim = len(data.feature_names)
out_dim = len(data.target_names)


KAN_width_table = []
KAN_grid_table = [3, 4, 5, 6, 10]
KAN_degree_table = [1, 2, 3, 4, 5]
KAN_lambda_table = np.linspace(0, 0.1, num=25)

for i in range(3, 11):
    current_width = [in_dim, i, out_dim]
    KAN_width_table.append(current_width)

KAN_width_table.append([in_dim, 16, out_dim])

for i in range(3, 9):
    for j in range(3, 9):
        current_width = [in_dim, i, j, out_dim]
        KAN_width_table.append(current_width)


for grid in KAN_grid_table:
    for degree in KAN_degree_table:
        for lam in KAN_lambda_table:
            for width in KAN_width_table:
                intro_line = "------------- start ---- grid: " + str(grid) + " deg: " + str(degree) + " lambda: " + str(lam) + " width: " + str(width) + " DT: " + string_datetime()
                write_to_file(intro_line)
                model, results, best_val_acc, last_i = trainKAN(dataset, steps=STEPS, KAN_width=width, KAN_grid=grid, KAN_degree=degree, KAN_lambda=lam, KAN_sparse_init=False)
                first_train_line = "normal train: " + str(last_i) + " steps; best val acc: " + str(best_val_acc)
                write_to_file(first_train_line)
                write_model_parameters(model)

                interpretable_line = "Turning interpretable ---" + string_datetime() + "---> formulas for each input:"
                write_to_file(interpretable_line)
                model.prune()
                model.auto_symbolic()
                write_model_formula(model)
                new_best_acc = torch.mean((torch.argmax(model(dataset['test_input']), dim=1) == dataset['test_label']).type(dtype)).item()
                line = "Validation accuracy after turning: " + str(new_best_acc)
                write_to_file(line)
                
                train_line = "Training interpretable ---" + string_datetime()
                write_to_file(train_line)
                try:
                    model, results, best_val_acc, last_i = trainKAN(dataset, steps=STEPS, KAN_lambda=lam, last_best_acc=new_best_acc, model=model)
                    second_train_line = "intepretable train: " + str(last_i) + " steps; best val acc: " + str(best_val_acc) + " DT: " + string_datetime()
                    write_to_file(second_train_line)
                except:
                    fail_line = "FAILED---" + string_datetime()
                    write_to_file(fail_line)


                

                
                            
