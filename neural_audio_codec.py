import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, stride=2)
        self.lin = nn.Linear(128*8, 512)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 128*8)
        return self.lin(x)

class Quantizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(512, 512)
        self.lin2 = nn.Linear(512, 16*32)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x.view(-1, 16, 32)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.convt1 = nn.ConvTranspose1d(16, 64, kernel_size=5, stride=2)
        self.convt2 = nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2)
        self.convt3 = nn.ConvTranspose1d(32, 1, kernel_size=5, stride=2)

    def forward(self, x):
        x = x.transpose(1, 2)  # Add channel dimension
        x = F.relu(self.convt1(x))
        x = F.relu(self.convt2(x))
        return torch.tanh(self.convt3(x))

model = nn.Sequential(Encoder(), Quantizer(), Decoder())
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Train...
for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
