from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Flatten, Dropout, Dense, LeakyReLU, Reshape
from tensorflow.keras.optimizers import Adam
from numpy import expand_dims
from numpy.random import randint, randn
from numpy import ones, zeros
import time

start_time = time.time()

# define the discriminator model
def define_discriminator(in_shape=(28,28,1)):
    model = Sequential()
    model.add(Conv2D(128, (3,3), strides=(2,2), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

# define the generator model
def define_generator(latent_dim):
    model = Sequential()
    model.add(Dense(128 * 7 * 7, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((7, 7, 128)))
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(1, (7, 7), activation='tanh', padding='same'))
    model.summary()
    return model

# define GAN model( stack generator model and discriminator model
def define_gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer = opt)
    return model


def load_dataset():
    (trainX, _), (_, _) = fashion_mnist.load_data()
    X = trainX[:30000]
    X = expand_dims(X, axis=-1).astype('float32')
    X = (X - 127.5) / 127.5
    return X


# select real sample
def generate_real_samples(dataset, n_samples):
    # choose random index
    idx = randint(0, dataset.shape[0], n_samples)
    # select image using random index
    X = dataset[idx]
    # genrate class lables
    y = ones((n_samples, 1))
    return X, y

# generate fake samples
def generate_fake_smaples(generator, latent_dim, n_samples):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    X = generator.predict(x_input)
    # create class lables
    y = zeros((n_samples, 1))
    return X, y

# generate points in latent space as input for generator
def generate_latent_points(latent_dim, n_samples):
    x_input = randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

# train GAN
def train(gen_model, dis_model,gan_model, dataset, latent_dim , epochs= 1, batch = 128 ):
    batch_per_epoch = int(dataset.shape[0]/ batch)
    half_batch = int( batch /2 )

    for i in range(epochs):

        for j in range(batch_per_epoch):

            X_real, y_real = generate_real_samples(dataset, half_batch)

            d_loss1, _ = dis_model.train_on_batch(X_real, y_real)

            X_fake, y_fake = generate_fake_smaples(gen_model,latent_dim, half_batch)

            d_loss2 ,_ = dis_model.train_on_batch(X_fake, y_fake)

            X_gan = generate_latent_points(latent_dim,batch)

            y_gan = ones((batch, 1))

            g_loss = gan_model.train_on_batch(X_gan, y_gan)

            print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f '  %
                  (i + 1, j + 1, batch_per_epoch, d_loss1, d_loss2, g_loss))

        gen_model.save('model/generator.h5')

# size of the latent space
latent_dim = 100
# create the discriminator
discriminator = define_discriminator()
# create the generator
generator = define_generator(latent_dim)
# create the gan
gan_model = define_gan(generator, discriminator)
# load image data
dataset = load_dataset()
# train model
train(generator, discriminator, gan_model, dataset, latent_dim)

print('-----%s seconds-----'%(time.time() - start_time))


















