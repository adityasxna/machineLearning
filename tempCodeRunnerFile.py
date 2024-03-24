import numpy as np
import matplotlib.pyplot as plt 

im = np.arange(10)
im = im[np.newaxis, :]
im = np.repeat(im, 10, axis=0)
plt.imshow(im, cmap='gray')