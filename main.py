import sklearn
from kan import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine, load_iris

PATIENCE_VALUE = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.get_default_dtype()
result_type = torch.long

def processDataset(data): #sklearn.datasets data
    y = data.target
    X = data.data
    X = sklearn.preprocessing.normalize(X)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.33) #random_state=7
    dataset = {}
    dataset['train_input'] = torch.from_numpy(np.array(X_train)).type(dtype).to(device)
    dataset['test_input'] = torch.from_numpy(np.array(X_test)).type(dtype).to(device)
    dataset['train_label'] = torch.from_numpy(np.array(y_train)).type(result_type).to(device)
    dataset['test_label'] = torch.from_numpy(np.array(y_test)).type(result_type).to(device)
    return dataset

def doKAN(dataset, steps, KAN_width, KAN_grid, KAN_degree, KAN_lambda, last_best_acc=0, model=None):
    def train_acc():
        return torch.mean((torch.argmax(model(dataset['train_input']), dim=1) == dataset['train_label']).type(dtype))

    def test_acc():
        return torch.mean((torch.argmax(model(dataset['test_input']), dim=1) == dataset['test_label']).type(dtype))

    if model is None:
        model = KAN(width=KAN_width, grid=KAN_grid, k=KAN_degree, device=device)

    final_results = {}
    final_results['train_acc'] = []
    final_results['test_acc'] = []
    best_test_acc = last_best_acc
    patience = PATIENCE_VALUE
    for i in range(steps):
        fit_results = model.fit(dataset, opt='LBFGS', steps=1, lamb=KAN_lambda, metrics=(train_acc, test_acc), loss_fn=torch.nn.CrossEntropyLoss())
        final_results['train_acc'].append(fit_results['train_acc'][0])
        final_results['test_acc'].append(fit_results['test_acc'][0])
        if fit_results['test_acc'][0] < best_test_acc:
            if patience == 0:
                return model, final_results, best_test_acc
            else:
                patience -= 1
        else:
            patience = PATIENCE_VALUE
            best_test_acc = fit_results['test_acc'][0]

    return model, final_results, best_test_acc

data = load_iris()
dataset = processDataset(data)

#model = KAN(width=[4,4,2,2,3], grid=3, k=0, device=device)
#results = model.fit(dataset, opt="LBFGS", steps=3, metrics=(train_acc, test_acc), loss_fn=torch.nn.CrossEntropyLoss())

model, results, best_test_acc = doKAN(dataset, 3, [4,4,2,3], 3, 2, 0.001)

print(results['train_acc'])
print(results['test_acc'])


model, results, best_test_acc = doKAN(dataset, 30, [4,4,2,3], 3, 2, 0.001, last_best_acc=best_test_acc, model=model)

print(results['train_acc'])
print(results['test_acc'])

print(torch.mean((torch.argmax(model(dataset['test_input']), dim=1) == dataset['test_label']).type(dtype)))
#model.plot(folder='./figures_pre')
exit(0)
model.prune()
model.auto_symbolic()
model.plot(folder='./figures_post')

formulas, params = model.symbolic_formula()
print(formulas)
