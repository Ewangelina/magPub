from kan import *
torch.set_default_dtype(torch.float64)

filepath = 'C:/Users/E D/Desktop/plots/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# create a KAN: 2D inputs, 1D output, and 5 hidden neurons. cubic spline (k=3), 5 grid intervals (grid=5).
model = KAN(width=[2,5,1], grid=3, k=3, seed=42, device=device)

from kan.utils import create_dataset
# create dataset f(x,y) = exp(sin(pi*x)+y^2)
f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
dataset = create_dataset(f, n_var=2, device=device)
dataset['train_input'].shape, dataset['train_label'].shape

# plot KAN at initialization
model(dataset['train_input']);
model.plot()


# train the model
model.fit(dataset, opt="LBFGS", steps=50, lamb=0.001);


model.plot()
filename = filepath + "hello_1.jpg"
plt.savefig(filename)

model = model.prune()
model.plot()
filename = filepath + "hello_2.jpg"
plt.savefig(filename)
