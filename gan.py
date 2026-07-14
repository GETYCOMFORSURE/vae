import torch
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt



# 1. data loader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # scales [0,1] -> [-1,1]
])

train_data = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform)

test_data = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform)

# shuffle + batch
train_loader = DataLoader(train_data, batch_size=32, shuffle=True) # returns a DataLoader object
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)



# 2. build model
# GAN: generator (noise -> fake image) + discriminator (image -> real/fake)

latent_dim = 100

# generator: latent noise -> 784, tanh to match [-1,1] data range
generator = nn.Sequential(
    nn.Linear(latent_dim, 128),
    nn.LeakyReLU(0.01),
    nn.Linear(128, 28*28),
    nn.Tanh()
)

# discriminator: image -> probability it's real
discriminator = nn.Sequential(
    nn.Linear(28*28, 128),
    nn.LeakyReLU(0.01),
    nn.Linear(128, 1),
    nn.Sigmoid()
)



# 3. loss function

# loss
criterion = nn.BCELoss()

# optimizer (corresponds to update parameters in numpy version)
discriminator_optim = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
generator_optim = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))



# 4. training loop
# outer loop over epochs, inner loop pulling batches out of train_loader.
# universal PyTorch training loop: zero_grad → loss → backward → step
# phase 1: train D. 
# phase 2: train G.

for epoch in range(100):
    for images, _ in train_loader:
        images = images.view(images.size(0), -1)
        
        noise = torch.randn(images.size(0), latent_dim)   # both phases
        fakes = generator(noise)                          # both phases
        real_labels = torch.ones(images.size(0), 1)       # both phases (target for G's lie in phase 2)
        fake_labels = torch.zeros(images.size(0), 1)      # phase 1 only
        
        # ===== PHASE 1: train D ===== 
        discriminator_optim.zero_grad()
        real_loss = criterion(discriminator(images), real_labels)
        fake_loss = criterion(discriminator(fakes.detach()), fake_labels) # detach = don't waste compute updating G's graph here
        d_loss = real_loss + fake_loss
        d_loss.backward()
        discriminator_optim.step()
        
        # ===== PHASE 2: train G =====         
        generator_optim.zero_grad()
        g_loss = criterion(discriminator(fakes), real_labels)
        g_loss.backward()
        generator_optim.step()

    if epoch % 10 == 0 or epoch == 99:
        print(f"Epoch {epoch}, D_loss: {d_loss.item():.4f}, "
              f"G_loss: {g_loss.item():.4f}")
