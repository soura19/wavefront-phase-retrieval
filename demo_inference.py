
#This is to check if the model is performing well in a completely new PSF

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift

#Define CNN architecture. Future modifications to network depth or kernel sizes can be done here
class ComaCNN(nn.Module):
    def __init__(self):
        super(ComaCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 40 * 40, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 40 * 40)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#load model
model = ComaCNN()
model.load_state_dict(torch.load('coma_cnn.pth', weights_only=True))
model.eval() #to denote evaluating and not training

#generate optical error
true_coef = np.random.uniform(-4.0, 4.0)

#Simulate optical aperture and wave propagation to the focal plane
N = 256
x = np.linspace(-1, 1, N)
y = np.linspace(-1, 1, N)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)
Theta = np.arctan2(Y, X)
pupil = R <= 1.0

coma_phase = true_coef * (3 * R**3 - 2 * R) * np.sin(Theta)
wavefront = pupil * np.exp(1j * coma_phase)
psf = np.abs(fftshift(fft2(wavefront, s=(1024, 1024))))**2

center = 512
zoom = 80
psf_zoom = psf[center-zoom:center+zoom, center-zoom:center+zoom]
final_image = np.log10(psf_zoom + 1).astype(np.float32)

#now to give this image to the AI, we must reshape it to what pytorch expects:
#(1 sample, 1 channel, 160 height, 160 width)
input_tensor = torch.tensor(final_image).unsqueeze(0).unsqueeze(0)

with torch.no_grad():
    prediction = model(input_tensor).item()

#results
plt.figure(figsize=(6, 6))
plt.imshow(final_image, cmap='inferno')
plt.colorbar(label='Log(Intensity)')
plt.title(f"AI Phase Retrieval Test\nTrue Error: {true_coef:.3f} | AI Guessed: {prediction:.3f}")
plt.axis('off')
plt.show()