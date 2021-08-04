#Standard libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import image
import glob

#Preprocessing libraries
import random
import pandas as pd
from astropy.visualization import astropy_mpl_style
from astropy.io import fits
from PIL import Image

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, Flatten, Reshape, Conv2D, MaxPooling2D, Activation, Dropout, BatchNormalization, Concatenate, Lambda, Conv2DTranspose
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.backend import set_image_data_format

#Testing libraries
from sklearn.model_selection import train_test_split
from scikitplot.metrics import plot_roc
from sklearn.metrics import confusion_matrix

#Augmentation
from augment import aug

#Random seed
seed = 1234

#Visuals
set_image_data_format('channels_last')
plt.style.use(astropy_mpl_style)

storagePATH = '/media/miron/ec129152-c1e2-41c2-a7a2-54cc3884cb13/'

def cropImage(image, size):
    x_dim, y_dim = image.shape
    xmin = int(0.5*(x_dim - size))
    xmax = int(0.5*(x_dim + size))
    ymin = int(0.5*(y_dim - size))
    ymax = int(0.5*(y_dim + size))

    return image[xmin:xmax, ymin:ymax]

def get_image(im_name, SCALE, mask):
    imagePATH = storagePATH + 'data/fits_512px/' + im_name
    im = fits.open(imagePATH)
    if mask == True:
        imCropped = cropImage(im[0].data, 256)
    else:
        imCropped = cropImage(im[0].data[0], 256)
    im.close()

    if SCALE == True:
        c = 1.0/np.log(1 + imCropped.max())
        imCropped = c*np.log(1 + imCropped)

    return imCropped[:,:,np.newaxis]


def get_data(whatDATA):
    #Import meta table.
    tablePATH = storagePATH + 'data/Tables/meta_p95_with_3sig_asym.csv'
    df = pd.read_csv(tablePATH)

    #Seperate classes (WITH, WITHOUT) features.
    noFeat = df.query('CONF == 0 or CONF == 1').reset_index(drop = True)
    noFeat = noFeat[noFeat.FEAT == 'N'].reset_index(drop = True)
    noFeat_name = noFeat[whatDATA + '_filename']

    withFeat = df.query('CONF == 3 or CONF == 4').reset_index(drop = True)
    withFeat = withFeat[withFeat.FEAT != 'N'].reset_index(drop = True)
    withFeat_name = withFeat[whatDATA + '_filename']

    #Get categories.
    categories = pd.Series(withFeat['FEAT']).unique()

    #Combine data.
    data = np.concatenate((withFeat_name, noFeat_name))

    #return TABLE
    return data, withFeat_name, noFeat_name

def test_data(whatDATA, SCALE = True, mask = False):
    #Import meta table.
    tablePATH = storagePATH + 'data/Tables/meta_p95_with_3sig_asym.csv'
    df = pd.read_csv(tablePATH)

    #Seperate classes (WITH, WITHOUT) features.
    noFeat = df.query('CONF == 0 or CONF == 1').reset_index(drop = True)
    noFeat = noFeat[noFeat.FEAT == 'N'].reset_index(drop = True)
    noFeat_name = noFeat[whatDATA + '_filename']

    withFeat = df.query('CONF == 3 or CONF == 4').reset_index(drop = True)
    withFeat = withFeat[withFeat.FEAT != 'N'].reset_index(drop = True)
    withFeat_name = withFeat[whatDATA + '_filename']

    #Get categories.
    categories = pd.Series(withFeat['FEAT']).unique()

    noFeatures = []
    target0 = []

    withFeatures = []
    target1 = []

    #get larger set
    if len(noFeat) >= len(withFeat):
        nb_sample = len(withFeat)
    else:
        nb_sample = len(noFeat)
   
    #noFeat = 0
    for n in range (1):
        for i in range (nb_sample):
            imagePATH = storagePATH + 'data/fits_512px/' + noFeat_name[i]
            im = fits.open(imagePATH)
            if mask == True:
                imCropped = cropImage(im[0].data, 256)
            else:
                imCropped = cropImage(im[0].data[0], 256)
                imCropped = imCropped[:,:,np.newaxis]

                #imCropped = 1.0 + imCropped
            noFeatures.append(imCropped)
            target0.append(0)
            im.close()
    
    #withFeat = 1
   
        for i in range (nb_sample):
            imagePATH = storagePATH + 'data/fits_512px/' + withFeat_name[i]
            im = fits.open(imagePATH)
            if mask == True:
                imCropped = cropImage(im[0].data, 256)
            else:
                imCropped = cropImage(im[0].data[0], 256)
                imCropped = imCropped[:,:,np.newaxis]
                #imCropped = 1.0 + imCropped
            withFeatures.append(imCropped)
            target1.append(1)
            im.close()

    #Concatenate ALL data.
    if SCALE == True:
        withFeatures = np.array(withFeatures)
        c1 = 1.0/withFeatures.max()
        noFeatures = np.array(noFeatures)
        c2 = 1.0/noFeatures.max()

        data = np.concatenate((c1*withFeatures, c2*noFeatures), axis = 0)
    else:
        data = np.concatenate((withFeatures, noFeatures), axis = 0)

    target = np.concatenate((target1, target0), axis = 0)

    #Add colour channel.
    #data = data[:,:,:,np.newaxis]

    #return IMAGES
    return data, target

def data_categories(whatDATA = 'threshold_3sig', SCALE = False, mask = False):
    #Import meta table.
    tablePATH = storagePATH + 'data/Tables/meta_p95_with_3sig_asym.csv'
    df = pd.read_csv(tablePATH)

    #Seperate classes (WITH, WITHOUT) features.
    noFeat = df.query('CONF == 0 or CONF == 1').reset_index(drop = True)
    noFeat = noFeat[noFeat.FEAT == 'N'].reset_index(drop = True)
    noFeat_name = noFeat[whatDATA + '_filename']

    shell = df.query('CONF == 3 or CONF == 4').reset_index(drop = True)
    shell = shell[shell.FEAT == 'S'].reset_index(drop = True)
    shell_name = shell[whatDATA + '_filename']
    
    arm = df.query('CONF == 3 or CONF == 4').reset_index(drop = True)
    arm = arm[arm.FEAT == 'A'].reset_index(drop = True)
    arm_name = arm[whatDATA + '_filename']


    noFeatures = []
    target0 = []

    shellFeatures = []
    target1 = []
    
    armFeatures = []
    target2 = []

    sizes = np.array([len(noFeat), len(shell), len(arm)])
    
    #get larger set
    if sizes.min() == len(noFeat):
        nb_sample = len(noFeat)
    elif sizes.min() == len(shell):
        nb_sample = len(shell)
    else:
        nb_samples = len(arm)
   
    #noFeat = 0.(arm)
    for n in range (1):
        for i in range (nb_sample):
            imagePATH = storagePATH + 'data/fits_512px/' + noFeat_name[i]
            im = fits.open(imagePATH)
            if mask == True:
                imCropped = cropImage(im[0].data, 256)
            else:
                imCropped = cropImage(im[0].data[0], 256)
                imCropped = imCropped[:,:,np.newaxis]

                #imCropped = 1.0 + imCropped
            noFeatures.append(imCropped/imCropped.max())
            target0.append(0)
            im.close()
    
    #withFeat = 1.(shell)
   
        for i in range (nb_sample):
            imagePATH = storagePATH + 'data/fits_512px/' + shell_name[i]
            im = fits.open(imagePATH)
            if mask == True:
                imCropped = cropImage(im[0].data, 256)
            else:
                imCropped = cropImage(im[0].data[0], 256)
                imCropped = imCropped[:,:,np.newaxis]
                #imCropped = 1.0 + imCropped
            shellFeatures.append(imCropped/imCropped.max())
            target1.append(1)
            im.close()
            
        for i in range (nb_sample):
            imagePATH = storagePATH + 'data/fits_512px/' + arm_name[i]
            im = fits.open(imagePATH)
            if mask == True:
                imCropped = cropImage(im[0].data, 256)
            else:
                imCropped = cropImage(im[0].data[0], 256)
                imCropped = imCropped[:,:,np.newaxis]
                #imCropped = 1.0 + imCropped
            armFeatures.append(imCropped/imCropped.max())
            target2.append(2)
            im.close()

    #Concatenate ALL data.
    if SCALE == True:
        withFeatures = np.array(withFeatures)
        #c1 = 1.0/withFeatures.max()
        noFeatures = np.array(noFeatures)
        #c2 = 1.0/noFeatures.max()

        data = np.concatenate((withFeatures, noFeatures), axis = 0)
    else:
        data = np.concatenate((noFeatures, shellFeatures, armFeatures), axis = 0)

    target = np.concatenate((target0, target1, target2), axis = 0)

    #Add colour channel.
    #data = data[:,:,:,np.newaxis]
    
    #Shuffle data
    perm = np.random.permutation(len(data))
    data = data[perm]
    target = target[perm]
    
    #return IMAGES
    return data, target


def test_data_all(whatDATA = 'threshold_3sig', SCALE = True, mask = False):
    #Import meta table.
    tablePATH = storagePATH + 'data/Tables/meta_p95_with_3sig_asym.csv'
    df = pd.read_csv(tablePATH)

    #Seperate classes (WITH, WITHOUT) features.
    noFeat = df.query('CONF == 3 or CONF == 4').reset_index(drop = True)
    noFeat = noFeat[noFeat.FEAT == 'A'].reset_index(drop = True)
    noFeat_name = noFeat[whatDATA + '_filename']

    withFeat = df.query('CONF == 3 or CONF == 4').reset_index(drop = True)
    withFeat = withFeat[withFeat.FEAT == 'S'].reset_index(drop = True)
    withFeat_name = withFeat[whatDATA + '_filename']

    #Get categories.
    categories = pd.Series(withFeat['FEAT']).unique()

    noFeatures = []
    target0 = []

    withFeatures = []
    target1 = []

    #get larger set
    if len(noFeat) >= len(withFeat):
        nb_sample = len(withFeat)
    else:
        nb_sample = len(noFeat)
   
    for n in range (1):
        for i in range (nb_sample):
            imagePATH = storagePATH + 'data/fits_512px/' + noFeat_name[i]
            im = fits.open(imagePATH)
            if mask == True:
                imCropped = cropImage(im[0].data, 256)
            else:
                imCropped = cropImage(im[0].data[0], 256)
                imCropped = aug(imCropped[:,:,np.newaxis])

                #imCropped = 1.0 + imCropped
            noFeatures.append(imCropped/imCropped.max())
            target0.append(0)
            im.close()
   
        for i in range (nb_sample):
            imagePATH = storagePATH + 'data/fits_512px/' + withFeat_name[i]
            im = fits.open(imagePATH)
            if mask == True:
                imCropped = cropImage(im[0].data, 256)
            else:
                imCropped = cropImage(im[0].data[0], 256)
                imCropped = aug(imCropped[:,:,np.newaxis])
                #imCropped = 1.0 + imCropped
            withFeatures.append(imCropped/imCropped.max())
            target1.append(1)
            im.close()

    #Concatenate ALL data.
    if SCALE == True:
        withFeatures = np.array(withFeatures)
        #c1 = 1.0/withFeatures.max()
        noFeatures = np.array(noFeatures)
        #c2 = 1.0/noFeatures.max()

        data = np.concatenate((withFeatures, noFeatures), axis = 0)
    else:
        data = np.concatenate((withFeatures, noFeatures), axis = 0)

    target = np.concatenate((target1, target0), axis = 0)

    #Add colour channel.
    #data = data[:,:,:,np.newaxis]

    #return IMAGES
    return withFeatures, target1, noFeatures, target0

'''
VAE SPECIFIC
'''
intermediate_dim = 250
shape = (256,256,1)

def sampling (args):
  # Unpack arguments
  z_mean, z_log_var = args

  # Get shape of random noise to sample
  epsilon = K.random_normal(shape=K.shape(z_mean))

  # Return samples from latent space p.d.f.
  return z_mean + K.exp(0.5 * z_log_var) * epsilon


def build_encoder (latent_dim, nb_categories=None):
  
    # Image input
    i = Input(shape=shape, name='input')
    inputs = i
    
    # Encoder architecture
    x = Conv2D(32, kernel_size = 3, strides = 2, activation = 'relu', padding = 'same')(i)
    x = Conv2D(48, kernel_size = 3, strides = 2, activation = 'relu', padding = 'same')(x)
    x = Conv2D(64, kernel_size = 3, strides = 2, activation = 'relu', padding = 'same')(x)
    x = Flatten()(x)
  
    x = Dense(128, activation = 'relu')(x)
    
    # Parametrise latent p.d.f.
    z_mean    = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # Sample from latent p.d.f.
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # List outputs
    outputs = [z_mean, z_log_var, z]

    # Return encoder model
    return Model(inputs, outputs, name='encoder')

def build_decoder (latent_dim, nb_categories=None):
  
    # Latent input
    i = Input(shape=(latent_dim,), name='z_sampling')
    inputs = i
    
    # Decoder architecture
    x = i
    x = Dense(units = 32*32*64, activation = 'relu')(x)
    x = Reshape(target_shape = (32,32,64))(x)
    x = Conv2DTranspose(64, kernel_size = 3, strides = 2, activation = 'relu', padding = 'same')(x)
    x = Conv2DTranspose(48, kernel_size = 3, strides = 2, activation = 'relu', padding = 'same')(x)
    x = Conv2DTranspose(32, kernel_size = 3, strides = 2, activation = 'relu', padding = 'same')(x)
    
    # Reshape to original image shape
    output = Conv2DTranspose(1, kernel_size = 3, activation = 'sigmoid', padding = 'same')(x)
    
    # Return decoder model
    return Model(inputs, output, name='decoder')


def build_vae (name, latent_dim, nb_categories=None):

    # Get encoder and decoder instances  
    encoder = build_encoder(latent_dim, nb_categories)
    decoder = build_decoder(latent_dim, nb_categories)

    # Chain together to get VAE
    i = encoder.inputs
    if len(i) == 1:
        i = i[0]
        pass
    z = encoder(i)[2]
    if nb_categories:
        z = [z, i[1]]
        pass
    o = decoder(z)

    # Return VAE model
    return Model(i, o, name=name)

from tensorflow.python.keras.losses import binary_crossentropy

def compile_vae (vae):

    # Get the latent p.d.f. mean and log-variance output layers from VAE encoder
    encoder   = vae.get_layer('encoder')
    z_log_var = encoder.get_layer('z_log_var').output
    z_mean    = encoder.get_layer('z_mean').output

    # Define reconstruction loss
    def reco_loss (y_true, y_pred):
        # Use binary cross-entropy loss
        reco_loss_value = binary_crossentropy(y_true, y_pred) 
        reco_loss_value = K.mean(reco_loss_value)#, axis=(1,2))
        return (reco_loss_value)

    # Define Kullback-Leibler loss with reference to encoder output layers
    def kl_loss (y_true, y_pred):
        kl_loss_value = 0.5 * (K.square(z_mean) + K.exp(z_log_var) - 1. - z_log_var)
        kl_loss_value = K.mean(kl_loss_value, axis=-1)
        return kl_loss_value

    # Define VAE loss
    def vae_loss (y_true, y_pred):
        return reco_loss(y_true, y_pred) + kl_loss(y_true, y_pred)

    vae.compile(optimizer='adam', loss=vae_loss, metrics=[reco_loss, kl_loss])
    return


