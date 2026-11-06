#Generate synthetic PSF dataset for CNN training
import numpy as np
from scipy.fft import fft2, fftshift
import time

#static optical grid
N = 256
x = np.linspace(-1, 1, N)
y = np.linspace(-1, 1, N)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)
Theta = np.arctan2(Y, X)
pupil = R <= 1.0

#parameters
pad_size = 1024
center = pad_size // 2
zoom = 80  #Final images will be 160x160
num_samples = 2000  # 2000 images only as I have an i3 processor 

#Pre-allocate memory using float32 to save RAM
images = np.zeros((num_samples, zoom*2, zoom*2), dtype=np.float32)
labels = np.zeros(num_samples, dtype=np.float32)

print(f"Starting generation of {num_samples} optical simulations...")
start_time = time.time()

#generate sample data:
for i in range(num_samples):
    #Generate a random coma coefficient between -4.0 and 4.0
    coef = np.random.uniform(-4.0, 4.0)
    
    #Compute Zernike phase distortion and Fraunhofer diffraction (PSF)
    coma_phase = coef * (3 * R**3 - 2 * R) * np.sin(Theta)
    wavefront = pupil * np.exp(1j * coma_phase)
    psf = np.abs(fftshift(fft2(wavefront, s=(pad_size, pad_size))))**2
    
    #Extract the zoomed section
    psf_zoom = psf[center-zoom:center+zoom, center-zoom:center+zoom]
    
    #Save to our arrays and apply log
    images[i] = np.log10(psf_zoom + 1).astype(np.float32)
    labels[i] = coef
    
    #Progress tracker
    if (i + 1) % 200 == 0:
        print(f"Generated {i + 1}/{num_samples} images...")

#Using savez_compressed keeps the file size small
np.savez_compressed('coma_dataset.npz', images=images, labels=labels)

end_time = time.time()
print(f"Success! Dataset saved to 'coma_dataset.npz' in {round(end_time - start_time, 1)} seconds.")