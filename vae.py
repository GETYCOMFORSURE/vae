!pip install torch

import torch
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

# 1. data loader
train_data = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transforms.ToTensor())

test_data = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transforms.ToTensor())

# shuffle + batch
train_loader = DataLoader(train_data, batch_size=32, shuffle=True) # returns a DataLoader object
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# 2. build model - define the architecture (what transforms input → output)

# VAE: encoder → μ, σ → sample once (z = μ + σ * ε) → decoder

# encoder outputs μ, σ
class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(28*28, 200)     # shared layer: input → hidden
        self.relu = nn.ReLU()
        self.fc_mu = nn.Linear(200, 10)      # branch 1: hidden → mean
        self.fc_logvar = nn.Linear(200, 10)  # branch 2: hidden → log-variance

    def forward(self, x):
        h = self.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

# sampled stochastically
def sampler(mu, logvar):
    eps = torch.randn_like(mu)     # ε ~ N(0,1), same shape as mu
    std = torch.exp(0.5 * logvar)  # convert logvar → std
    z = mu + std * eps
    return z

# decoder
# two layers: ReLU in between, sigmoid at the end.
class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(10, 200)
        self.fc2 = nn.Linear(200, 28*28)
        self.sig = nn.Sigmoid()

    def forward(self, y):
        h = self.relu(self.fc1(y))
        a = self.fc2(h)
        b = self.sig(a)
        return b

# 3. loss function
# loss
criterion = nn.BCELoss(reduction='sum')

# optimizer (corresponds to update parameters in numpy version)
encoder = Encoder()
decoder = Decoder()

optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001) # 'lr' is adjustable

# 4. traininig loop

# training loop
loss_history = []   
rec_history = []    
kl_history = [] 

for epoch in range(100):
    epoch_rec_loss = 0.0
    epoch_kl_loss = 0.0
    n_samples = 0

    for images, _ in train_loader:
        images = images.view(images.size(0), -1)
        optimizer.zero_grad()
        mu, logvar = encoder(images)
        z = sampler(mu, logvar)
        reconstructed = decoder(z)
        rec_loss = criterion(reconstructed, images)
        kl_loss = -0.5 * torch.sum(1 + logvar - torch.square(mu) - torch.exp(logvar))
        loss = rec_loss + kl_loss
        loss.backward()
        optimizer.step()

        # BCELoss/KL above use reduction='sum', i.e. summed over the whole
        # batch. Accumulate those sums, then divide by the number of
        # samples seen this epoch to get a per-example average that's
        # comparable across epochs (batch-to-batch noise averages out).
        epoch_rec_loss += rec_loss.item()
        epoch_kl_loss += kl_loss.item()
        n_samples += images.size(0)

    avg_rec_loss = epoch_rec_loss / n_samples
    avg_kl_loss = epoch_kl_loss / n_samples
    avg_loss = avg_rec_loss + avg_kl_loss

    loss_history.append(avg_loss)
    rec_history.append(avg_rec_loss)
    kl_history.append(avg_kl_loss)

    if epoch % 10 == 0 or epoch == 99:
        print(f"Epoch {epoch}, Avg Loss: {avg_loss:.4f} "
              f"(Rec: {avg_rec_loss:.4f}, KL: {avg_kl_loss:.4f})")

# plot the per-epoch average loss to confirm the steady decrease
plt.figure(figsize=(6, 4))
plt.plot(loss_history, label="Total loss")
plt.plot(rec_history, label="Reconstruction loss")
plt.plot(kl_history, label="KL loss")
plt.xlabel("Epoch")
plt.ylabel("Avg loss per sample")
plt.title("VAE training loss")
plt.legend()
plt.tight_layout()
plt.show()

# 5. evaluation

encoder.eval()
decoder.eval()

with torch.no_grad():  # don't track gradients, just looking not training
    images, labels = next(iter(test_loader))
    images_flat = images.view(images.size(0), -1)

    mu, logvar = encoder(images_flat)
    z = sampler(mu, logvar)
    reconstructed = decoder(z)
    reconstructed = reconstructed.view(-1, 1, 28, 28)  # reshape back to image shape

    fig, axes = plt.subplots(2, 8, figsize=(12, 3))
    for i in range(8):
        axes[0, i].imshow(images[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title('Original')

        axes[1, i].imshow(reconstructed[i].squeeze(), cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title('Reconstructed')

    plt.tight_layout()
    plt.show()
