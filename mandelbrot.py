import numpy as np
import matplotlib.pyplot as plt

# Parameters for Mandelbrot set
width = 1000
height = 1000
x_min, x_max = -2.5, 1
y_min, y_max = -1, 1
max_iter = 30

def generate_mandelbrot(width, height, x_min, x_max, y_min, y_max, max_iter):
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    X, Y = np.meshgrid(x, y)
    c = X + 1j * Y
    z = np.zeros_like(c)
    img = np.zeros((height, width))

    for i in range(max_iter):
        mask = np.abs(z) < 2
        z[mask] = z[mask] ** 2 + c[mask]
        img += mask

    return img

def visualize_mandelbrot(img):
    plt.imshow(np.log(img), cmap='hot', extent=[-2.5, 1, -1, 1])
    plt.colorbar()
    plt.title('Mandelbrot Set')
    plt.xlabel('Re')
    plt.ylabel('Im')
    plt.show()



# Generate the Mandelbrot set
# mandelbrot_img = generate_mandelbrot(width, height, x_min, x_max, y_min, y_max, max_iter)

# Visualize the Mandelbrot set
# visualize_mandelbrot(mandelbrot_img)
