from tensorflow.keras.models import load_model

from numpy.random import randn
from numpy import zeros
from matplotlib import pyplot

# generate points in latent space as input for generator
def generate_latent_points(latent_dim, n_samples):
    x_input = randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input
#
def show_plot(generated_img, n):
    for i in range(n):
        pyplot.subplot(n,n, 1+ i)
        pyplot.axis('off')
        pyplot.imshow(generated_img[i, :, :, 0], cmap= 'gray_r')
    pyplot.show()


model= load_model('model/generator.h5')

latent_points  = generate_latent_points( 100, 100)

X = model.predict(latent_points)

show_plot(X,5)

