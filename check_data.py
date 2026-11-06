#This is to check whether the created dataset works or not. There's no need to run it unless you're in doubt about the proper creation of the dataset.

import numpy as np

data = np.load('coma_dataset.npz')

images = data['images']
labels = data['labels']

print("Dataset successfully loaded!")
print(f"Images shape: {images.shape} (Should be 2000, 160, 160)")
print(f"Labels shape: {labels.shape} (Should be 2000,)")
print(f"First 5 coma coefficients: {labels[:5]}")