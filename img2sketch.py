from PIL import Image
import numpy as np

im = Image.open('pink.png')
im = im.convert("L")

data = np.array(im)
grad_x, grad_y = np.gradient(data)
print(grad_x.shape, grad_y.shape)
depth = 10
grad_x /= 100/depth
grad_y /= 100/depth
eLevation = np.pi / 2.2
azimuth = np.pi / 4

dx = np.cos(eLevation) * np.cos(azimuth)
dy = np.cos(eLevation) * np.sin(azimuth)
dz = np.sin(eLevation)
A = np.sqrt(grad_x ** 2 + grad_y ** 2 + 1.)
uni_x = grad_x / A
uni_y = grad_y / A
uni_z = 1. / A
b = 255 * (dx * uni_x + dy * uni_y + dz * uni_z)
res = Image.fromarray(b.clip(0, 255).astype('uint8'))
res.save('deal-pink.png')


