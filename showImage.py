import numpy as np
from PIL import Image
#from sklearn.datasets import load_digits
import matplotlib.pyplot as plt 



img_array = np.load('accuracy.npy') 

plt.plot(img_array)
plt.show()