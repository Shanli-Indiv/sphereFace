import numpy as np
from PIL import Image
#from sklearn.datasets import load_digits
import matplotlib.pyplot as plt 



img_array = np.load('loss.npy') 
print(img_array)
print(img_array.shape)

#img_array = img_array.reshape(200,140)
print(img_array.shape)

# im = Image.fromarray(img_array, 'L')
# this might fail if `img_array` contains a data type that is not supported by PIL,
# in which case you could try casting it to a different dtype e.g.:
# im = Image.fromarray(img_array.astype(np.uint8))
# im.show()
# plt.plot(img_array, np.zeros_like(img_array) + 0, 'x')
plt.plot(img_array)


plt.show()

#plt.imshow(img_array, cmap='gray')


# digits = img_array
# print(digits.data.shape)
# plt.gray() 
# plt.matshow(digits.images[0]) 
# plt.show() 