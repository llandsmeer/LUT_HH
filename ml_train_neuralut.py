import numpy
import torch
import numpy as np
import matplotlib.pyplot as plt
from neuralut_mnist_model import MnistNeqModel

config = {
        "hidden_layers": [12, 12, 12],
        "input_length": 2,
        "output_length": 1,
        "input_bitwidth": 12,
        "hidden_bitwidth": 12,
        "output_bitwidth": 12,
        "input_fanin": 2,
        "hidden_fanin": 6,
        "output_fanin": 6,
        "width_n": 16,
        "batch_size": 12,
        "cuda": True
        }



f = numpy.load('./dataset/h_next.npz')
i, v, g, o = f['i'], f['v'], f['g'], f['o']

print(i.shape, v.shape, g.shape, o.shape)

inp = numpy.stack([v, g]).T / (1<<12)
# out = o / (1<<12)
# out = (o - g)
out = o / (1 << 12)
# out = out / out.max()

inp = inp[::1000]
out = out[::1000]

X = torch.from_numpy(inp).float().reshape(-1, 2)
Y = torch.from_numpy(out).float().reshape(-1, 1)

model = MnistNeqModel(config)


Xcuda = X.to('cuda')
Ycuda = Y.to('cuda')
model = model.to('cuda')

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()
for epoch in range(5000):
    try:
        optimizer.zero_grad()
        outputs = model(Xcuda)
        loss = criterion(outputs, Ycuda)
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch+1}], Loss: {loss.item():.4f}')
    except:
        with torch.no_grad():
            y = model(Xcuda).cpu()

        yy = np.arange(Y.min(), Y.max())
        plt.plot(yy, yy-.5, color='black')
        plt.plot(yy, yy+.5, color='black')
        Yrnd = np.round(Y)
        Yrnd = Y
        # m = np.round(y) == Yrnd
        m = np.abs(y - Yrnd) < .5
        plt.scatter(Yrnd[m], y[m], alpha=0.5, color='black')
        plt.scatter(Yrnd[~m], y[~m], alpha=0.5, color='red')
        plt.axis('equal')
        plt.show()

torch.save(model.state_dict(), 'h_next.pth')

with torch.no_grad():
    y = model(Xcuda).cpu()

yy = np.arange(Y.min(), Y.max())
plt.plot(yy, yy-.5, color='black')
plt.plot(yy, yy+.5, color='black')
Yrnd = np.round(Y)
Yrnd = Y
# m = np.round(y) == Yrnd
m = np.abs(y - Yrnd) < .5
plt.scatter(Yrnd[m], y[m], alpha=0.5, color='black')
plt.scatter(Yrnd[~m], y[~m], alpha=0.5, color='red')
plt.axis('equal')
plt.show()
