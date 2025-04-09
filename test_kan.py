import sklearn
from kan import *
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.datasets import load_breast_cancer, load_wine, load_iris

PATIENCE_VALUE = 3
TOLERANCE_AMOUNT = 0

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
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

# create a KAN: 2D inputs, 1D output, and 5 hidden neurons. cubic spline (k=3), 5 grid intervals (grid=5).
model = KAN(width=[1,1], grid=3, k=3, seed=2, device=device)
dataset = create_dataset(wierd_fun, n_var=1, device=device)
print(wierd_fun(dataset['test_input'][0:10]))

# train the model
model.fit(dataset, opt="LBFGS", steps=2);

model.plot()
plt.show()

model.suggest_symbolic(0,0,0)

print(SYMBOLIC_LIB.keys())
add_symbolic('WF', wierd_fun, c=1)
model.suggest_symbolic(0,0,0)