import sklearn
from kan import *
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.datasets import load_breast_cancer, load_wine, load_iris

filepath = 'C:/Users/E D/Desktop/plots/'
device = 'cpu' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
dtype = torch.get_default_dtype()
result_type = torch.long

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

model = KAN(width=[1,1], grid=3, k=3, device=device)
dataset = create_dataset(wierd_fun, n_var=1, device=device)
print("Przykładowe dane -------- ")
print(dataset['test_input'][0:10])
print(wierd_fun(dataset['test_input'][0:10]))
print(len(dataset['train_input']))

model.fit(dataset, opt="LBFGS", steps=1);

model.plot()
#plt.show()
filename = filepath + "1.jpg"
plt.savefig(filename)

print("Bez nowej funkcji -------- ")
model.suggest_symbolic(0,0,0)

print("Dodanie nowej funkcji -------- ")
add_symbolic('WF', wierd_fun)
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
