import sklearn
from kan import *
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.datasets import load_breast_cancer, load_wine, load_iris

PATIENCE_VALUE = 10
TOLERANCE_AMOUNT = 0
STEPS = 1
BATCH_SIZE = -1
DELAY = 0

filepath = 'C:/Users/E D/Desktop/plots/'
device = 'cpu' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
dtype = torch.get_default_dtype()
result_type = torch.long

def write_to_file(line):
    return

def plot_model(filename, model, x_min=-1, x_max=1, n_points=100):
    global filepath
    
    X = np.linspace(x_min, x_max, n_points).reshape(-1, 1)
    X = torch.from_numpy(np.array(X)).type(dtype).to(device)
    Y = model(X)
    plt.plot(X.detach().numpy(), Y.detach().numpy())
    plt.savefig(filename)
    plt.clf()

    

#wierd_fun = lambda x: x**2

def wierd_fun(inp):   
    ret = []
    for el in inp:
        if el.dim() == 0:
            if el.item() > 0:
                ret.append(1)
            else:
                ret.append(-1)
        else:
            ret.append(wierd_fun(el))
    return torch.from_numpy(np.array(ret)).type(dtype).to(device)

WF = lambda x: torch.sign(x)
#Boolean value of Tensor with more than one value is ambiguous


dataset = create_dataset(WF, n_var=1, device=device)

#model = KAN(width=[2,1,1], grid=3, k=3, seed=1, device=device)
#f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
#dataset = create_dataset(f, n_var=2, device=device)
#model.fit(dataset, opt="LBFGS", steps=20)
#print("train acc:", torch.mean((torch.argmax(model(dataset['train_input']), dim=1) == dataset['train_label']).type(dtype)))
#print("test acc:", torch.mean((torch.argmax(model(dataset['test_input']), dim=1) == dataset['test_label']).type(dtype)))

print("Przykładowe dane -------- ")
print(dataset['test_input'][0:10])
print(WF(dataset['test_input'][0:10]))
print(len(dataset['train_input']))

model = KAN(width=[1,1], grid=3, k=3, device=device)
model.fit(dataset, opt="LBFGS", steps=1)

model.plot()
#plt.show()
filename = filepath + "1.jpg"
plt.savefig(filename)
plt.clf()
filename = filepath + "1a.jpg"
plot_model(filename, model)

print("Po większej ilości kroków -------- ")
model.fit(dataset, opt="LBFGS", steps=20)

model.plot()
#plt.show()
filename = filepath + "11.jpg"
plt.savefig(filename)
plt.clf()
filename = filepath + "11a.jpg"
plot_model(filename, model)

print("Bez nowej funkcji -------- ")
model.suggest_symbolic(0,0,0)

print("Dodanie nowej funkcji -------- ")
add_symbolic('WF', WF)
print(SYMBOLIC_LIB.keys())

print("Z nową funkcją -------- ")
model.suggest_symbolic(0,0,0)


print("Wzór -------- ")
#model.fix_symbolic(0,0,0,'WF',fit_params_bool=True)
model.auto_symbolic(lib=['WF'])
line = ""
formulas = model.symbolic_formula()[0]
for form in formulas:
    line += str(ex_round(form, 4)) + "\n"
print(line[:-1])

model.plot()
#plt.show()
filename = filepath + "2.jpg"
plt.savefig(filename)
plt.clf()
filename = filepath + "2a.jpg"
plot_model(filename, model)
