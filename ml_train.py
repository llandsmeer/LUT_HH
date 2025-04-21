import numpy
import torch
import numpy as np
import matplotlib.pyplot as plt

f = numpy.load('./dataset/h_next.npz')
i, v, g, o = f['i'], f['v'], f['g'], f['o']

print(i.shape, v.shape, g.shape, o.shape)

inp = numpy.stack([v, g]).T / (1<<12)
# out = o / (1<<12)
out = (o - g)
# out = out / out.max()

inp = inp[::50]
out = out[::50]

X = torch.from_numpy(inp).float().reshape(-1, 2)
Y = torch.from_numpy(out).float().reshape(-1, 1)

model = torch.nn.Sequential(
    torch.nn.Linear(2, 10),
    torch.nn.Tanh(),
    torch.nn.Linear(10, 6),
    torch.nn.Tanh(),
    torch.nn.Linear(6, 6),
    torch.nn.Tanh(),
    torch.nn.Linear(6, 6),
    torch.nn.Tanh(),
    torch.nn.Linear(6, 1)
)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()
for epoch in range(5000):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, Y)
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}], Loss: {loss.item():.4f}')

with torch.no_grad():
    y = model(X)

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
