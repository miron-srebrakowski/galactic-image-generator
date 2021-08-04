#Dataset
from utilsGAN import *

#Standatd libs
import matplotlib.pyplot as plt
import numpy as np
import time

#Model creation libs
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout, Embedding, Concatenate, BatchNormalization, ReLU

from augment import aug

#PARAMS
nb_epochs = 10000
batch_size = 64
train_size = 2000
latent_dim = 100
save_gap = 100


#DISCRIMINATOR
def make_discriminator(input_shape = (256,256,1), nb_classes = 3):
    input_label = Input(shape = (1,))

    #Embed label
    li = Embedding(nb_classes, 50)(input_label)

    #Scale up to image dimensions
    nb_nodes = input_shape[0] * input_shape[1]
    li = Dense(nb_nodes)(li)

    #Reshape as additional channel
    li = Reshape((input_shape[0], input_shape[1], 1))(li)

    #Merge embedded label with image
    input_image = Input(shape = input_shape)
    merge = Concatenate()([input_image, li])

    convargs = dict(kernel_size = (3,3), strides = (2,2), padding = 'same')

    x = Conv2D(32, **convargs)(merge)
    x = LeakyReLU(alpha = 0.2)(x)

    x = Conv2D(64, **convargs)(x)
    x = LeakyReLU(alpha = 0.2)(x)

    x = Conv2D(128, **convargs)(x)
    x = LeakyReLU(alpha = 0.2)(x)

    x = Conv2D(256, **convargs)(x)
    x = LeakyReLU(alpha = 0.2)(x)

    x = Conv2D(512, **convargs)(x)
    x = LeakyReLU(alpha = 0.2)(x)

    x = Flatten()(x)
    x = Dropout(0.4)(x)

    output = Dense(1, activation = 'sigmoid')(x)
    model = Model([input_image, input_label], output)

    optimizer = Adam(lr = 0.0002, beta_1 = 0.5)
    model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

    return model

#GENERATOR
def make_generator(latent_dim, nb_classes = 3):
    input_label = Input(shape = (1,))

    #Embed label
    li = Embedding(nb_classes, 50)(input_label)

    #Scale to low res image dimensions
    nb_nodes = 8*8
    li = Dense(nb_nodes)(li)

    #Reshape as additional channel
    li = Reshape((8,8,1))(li)

    input_vector = Input(shape = (latent_dim,))

    #Foundation for low res images (128 7x7)
    nb_nodes = 8*8*512
    x = Dense(nb_nodes)(input_vector)
    x = LeakyReLU(alpha = 0.2)(x)
    x = Reshape((8,8,512))(x)

    #Merge embedded label with low res images
    merge = Concatenate()([x, li])

    convargs = dict(kernel_size = (3,3), strides = (2,2), padding = 'same')

    x = Conv2DTranspose(512, **convargs)(merge)
    x = LeakyReLU(alpha = 0.2)(x)

    x = Conv2DTranspose(256, **convargs)(x)
    x = LeakyReLU(alpha = 0.2)(x)

    x = Conv2DTranspose(128, **convargs)(x)
    x = LeakyReLU(alpha = 0.2)(x)

    x = Conv2DTranspose(64, **convargs)(x)
    x = LeakyReLU(alpha = 0.2)(x)

    x = Conv2DTranspose(32, **convargs)(x)
    x = LeakyReLU(alpha = 0.2)(x)

    output = Conv2D(1, (7,7), activation = 'tanh', padding = 'same')(x)
    model = Model([input_vector, input_label], output)

    return model

#C-GAN
def make_gan(g_model, d_model):
    #Freeze discriminator weights.
    d_model.trainable = False

    #Get generator inputs
    g_noise, g_label = g_model.input

    #Get generator output
    g_output = g_model.output

    #Connect both parts
    gan_output = d_model([g_output, g_label])
    model = Model([g_noise, g_label], gan_output)

    optimizer = Adam(lr = 0.0002, beta_1 = 0.5)
    model.compile(loss = 'binary_crossentropy', optimizer = optimizer)

    return model

#LOAD AND SCALE DATA
def load_real_samples():
    images = np.load('data/images.npy')
    labels = np.load('data/labels.npy')

    #Add colour dimension
    X = images.astype('float32')

    #Scale from [0,1] to [-1,1]
    X = (X*2) - 1

    return [X, labels]

#GENERATE REAL IMAGE BATCH
def generate_real_batch(dataset, nb_samples):
    images, labels = dataset

    ix = np.random.randint(0, images.shape[0], nb_samples)

    X = []
    train_labels = np.zeros(len(ix))

    #Augment selected
    for i in range (len(ix)):
        X.append(aug(images[ix[i]]))
        train_labels[i] = labels[ix[i]]

    #Remove used
    new_data = []
    new_labels = []
    for i in range (len(images)):
        if i not in ix:
            new_data.append(images[i])
            new_labels.append(labels[i])

    dataset = [np.array(new_data), np.array(new_labels)]

    #Add noise
    #X = add_noise(X)

    #Get class label (1 = real image)
    y = np.ones((nb_samples, 1))

    return [np.array(X), train_labels], y, dataset

#SAMPLE LATENS SPACE FOR GENERATOR INPUT
def generate_latent_points(latent_dim, nb_samples, nb_classes = 2):
    x_input = np.random.randn(latent_dim * nb_samples)
    labels = np.random.randint(0, nb_classes, nb_samples)

    #Reshep into input shape batch
    z_input = x_input.reshape(nb_samples, latent_dim)

    return [z_input, labels]

#GENERATE FAKE IMAGE BATCH
def generate_fake_batch(generator, latent_dim, nb_samples):
    z_input, labels_input = generate_latent_points(latent_dim, nb_samples)

    #Get fake images
    images = generator.predict([z_input, labels_input])

    #Add noise
    #images = add_noise(images)

    #Get class label (0 = fake/generated images)
    y = np.zeros((nb_samples, 1))

    return [images, labels_input], y

#TRAIN GAN
def train(g_model, d_model, gan_model, dataset, latent_dim, nb_epochs, batch_size, train_size):
    half_batch = int(batch_size/2)
    nb_iter = int(train_size/half_batch)

    epochs = []
    d1_record = []
    d2_record = []
    g_record = []
    conf_record = []

    full_data = np.copy(dataset[0])
    full_labels = np.copy(dataset[1])

    #Manually enumerate epochs
    for i in range (nb_epochs):
        start = time.time()

        #Refill training data
        dataset = [full_data, full_labels]
        iteration = 0

        while iteration < nb_iter:
            #Refill training data
            if len(dataset[0]) < half_batch:
                dataset = [full_data, full_labels]

            #Get real samples
            [X_real, labels_real], y_real, dataset = generate_real_batch(dataset, half_batch)

            #Update discriminator weights
            d_loss1, _ = d_model.train_on_batch([X_real, labels_real], y_real)

            #Get fake samples
            [X_fake, labels_fake], y_fake = generate_fake_batch(g_model, latent_dim, half_batch)

            #Update discriminator weights
            d_loss2, _ = d_model.train_on_batch([X_fake, labels_fake], y_fake)

            #Get latent points (generator input) and 'false' labels
            [z_inputs, labels_inputs] = generate_latent_points(latent_dim, batch_size)
            y_gan = np.ones((batch_size, 1))

            #Update generator (via discriminator error)
            g_loss = gan_model.train_on_batch([z_inputs, labels_inputs], y_gan)

            iteration += 1

        end = time.time() - start
        #Display epoch loss
        print('Epoch %d/%d, %.1f sec, d1=%.3f, d2=%.3f, g=%.3f' % (i+1, nb_epochs, end, d_loss1, d_loss2, g_loss))

        #Record metrics
        epochs.append(i+1)
        d1_record.append(d_loss1)
        d2_record.append(d_loss2)
        g_record.append(g_loss)

        #Save generator every 10 epochs
        if (i+1) % save_gap == 0:
            g_model.save('models/gan_generator_' + str(i+1) +'.h5')
            print('Model saved!')

    return epochs, [d1_record, d2_record, g_record, conf_record]

def main():

    discriminator = make_discriminator()
    generator = make_generator(latent_dim)
    gan_model = make_gan(generator, discriminator)

    dataset = load_real_samples()

    epochs, metrics = train(generator, discriminator, gan_model, dataset, latent_dim, nb_epochs, batch_size, train_size)
