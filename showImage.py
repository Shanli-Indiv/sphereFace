import numpy as np
from PIL import Image

img_array = np.load('loss.npy')
print(img_array)
print(img_array.shape)

im = Image.fromarray(img_array, 'RGB')
# this might fail if `img_array` contains a data type that is not supported by PIL,
# in which case you could try casting it to a different dtype e.g.:
# im = Image.fromarray(img_array.astype(np.uint8))


im.show()