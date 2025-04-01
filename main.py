if False:
    from kan import KAN
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_moons
    import torch
    import numpy as np
    from kan import utils as ku

    dtype = torch.get_default_dtype()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = {}
    train_input, train_label = make_moons(n_samples=1000, shuffle=True, noise=0.1, random_state=None)
    test_input, test_label = make_moons(n_samples=1000, shuffle=True, noise=0.1, random_state=None)

    dataset['train_input'] = torch.from_numpy(train_input).type(dtype).to(device)
    dataset['test_input'] = torch.from_numpy(test_input).type(dtype).to(device)
    dataset['train_label'] = torch.from_numpy(train_label).type(torch.long).to(device)
    dataset['test_label'] = torch.from_numpy(test_label).type(torch.long).to(device)

    X = dataset['train_input']
    y = dataset['train_label']
    plt.scatter(X[:,0].cpu().detach().numpy(), X[:,1].cpu().detach().numpy(), c=y[:].cpu().detach().numpy())
    model = KAN(width=[2,2], grid=3, k=0, seed=2024, device=device)

    def train_acc():
        return torch.mean((torch.argmax(model(dataset['train_input']), dim=1) == dataset['train_label']).type(dtype))

    def test_acc():
        return torch.mean((torch.argmax(model(dataset['test_input']), dim=1) == dataset['test_label']).type(dtype))

    results = model.fit(dataset, opt="LBFGS", steps=20, metrics=(train_acc, test_acc), loss_fn=torch.nn.CrossEntropyLoss());
    lib = ['x','x^2','x^3','x^4','exp','log','sqrt','tanh','sin','abs']
    model.auto_symbolic(lib=lib)
    coss = model.symbolic_formula()
    print(coss)
    formula1, formula2 = coss[0]

    ku.ex_round(formula1, 4)
    print(formula1)
    # how accurate is this formula?
    def acc(formula1, formula2, X, y):
        batch = X.shape[0]
        correct = 0
        for i in range(batch):
            logit1 = np.array(formula1.subs('x_1', X[i,0]).subs('x_2', X[i,1])).astype(np.float64)
            logit2 = np.array(formula2.subs('x_1', X[i,0]).subs('x_2', X[i,1])).astype(np.float64)
            correct += (logit2 > logit1) == y[i]
        return correct/batch

    print('train acc of the formula:', acc(formula1, formula2, dataset['train_input'], dataset['train_label']))
    print('test acc of the formula:', acc(formula1, formula2, dataset['test_input'], dataset['test_label']))




    exit(0)



from kan import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine, load_iris

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
dtype = torch.get_default_dtype()
print(dtype)

data = load_iris()
labels = data.target
exes = data.data

choice = np.random.choice(range(labels.shape[0]), size=(int(labels.shape[0]*0.8),), replace=False)    
ind = np.zeros(labels.shape[0], dtype=bool)
ind[choice] = True
train_exes = []
test_exes = []
train_labels = []
test_labels = []
i = 0
for el in ind:
    if el:
        train_labels.append(labels[i])
        train_exes.append(exes[i])
    else:
        test_labels.append(labels[i])
        test_exes.append(exes[i])
    i += 1

dataset = {}
dataset['train_input'] = torch.from_numpy(np.array(train_exes)).type(dtype).to(device)
dataset['test_input'] = torch.from_numpy(np.array(test_exes)).type(dtype).to(device)
dataset['train_label'] = torch.from_numpy(np.array(train_labels)).type(torch.long).to(device)
dataset['test_label'] = torch.from_numpy(np.array(test_labels)).type(torch.long).to(device)

model = KAN(width=[4,4,2,2,3], grid=3, k=3, device=device)

def train_acc():
    return torch.mean((torch.argmax(model(dataset['train_input']), dim=1) == dataset['train_label']).type(dtype))

def test_acc():
    return torch.mean((torch.argmax(model(dataset['test_input']), dim=1) == dataset['test_label']).type(dtype))

results = model.fit(dataset, opt="LBFGS", steps=3, metrics=(train_acc, test_acc), loss_fn=torch.nn.CrossEntropyLoss())

print(results['train_acc'][-1])
print(results['test_acc'][-1])
#model.plot(folder='./figures_pre')
model.prune()
model.auto_symbolic()
model.plot(folder='./figures_post')

formulas, params = model.symbolic_formula()
print(formulas)
