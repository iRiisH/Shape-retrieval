import numpy as np
import matplotlib.pyplot as plt
from scipy import misc, signal 

def gabor_fn(sigma, theta, Lambda, psi, gamma):
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    # Bounding box
    nstds = 5 # Number of standard deviation sigma
    xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta)))
    xmax = np.ceil(max(1, xmax))
    ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta)))
    ymax = np.ceil(max(1, ymax))
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    # Rotation 
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)
    return gb


def convolve():
	gb = gabor_fn(2, 0.5, 0.4, 0., 1.)	
	print(gb.shape)
	# gb = gabor_fn(3., 0.5, 0.43, 0., 1.)
	img = misc.imread('views_8.jpg', 'L')
	res = signal.convolve(img, gb)
	return res


#gb = gabor_fn(100.,  0.5, 200., 0., 1.)	
#plt.imshow(gb)
#plt.colorbar()
#plt.show()
res = convolve()
print(np.max(res))
res /= 1000.
plt.imshow(res, clim=(0.0, 1.))
plt.colorbar()
plt.show()
