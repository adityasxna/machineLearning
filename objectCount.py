import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('bitss.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray')