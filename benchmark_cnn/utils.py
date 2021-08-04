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

#Model creation libraries
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, Flatten, Reshape, Conv2D, MaxPooling2D, Activation, Dropout, BatchNormalization
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.backend import set_image_data_format

#Testing libraries
from sklearn.model_selection import train_test_split
from scikitplot.metrics import plot_roc
from sklearn.metrics import confusion_matrix

#Augmentation
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from augment import aug

#Random seed
seed = 1234

#Visuals
set_image_data_format('channels_last')
plt.style.use(astropy_mpl_style)

storagePATH = '/media/miron/ec129152-c1e2-41c2-a7a2-54cc3884cb13'

def cropImage(image, size):
    x_dim, y_dim = image.shape
    xmin = int(0.5*(x_dim - size))
    xmax = int(0.5*(x_dim + size))
    ymin = int(0.5*(y_dim - size))
    ymax = int(0.5*(y_dim + size))
    
    return image[xmin:xmax, ymin:ymax]

def get_image(im_name, SCALE, mask):
    imagePATH = storagePATH + '/data/fits_512px/' + im_name
    im = fits.open(imagePATH)
    if mask == True:
        imCropped = cropImage(im[0].data, 256)
    else:
        imCropped = cropImage(im[0].data[0], 256)
    im.close()
    
    if SCALE == True:
        c = imCropped.max()/np.log(1 + imCropped.max())
        imCropped = c*np.log(1 + imCropped)
        
    return imCropped[:,:,np.newaxis]

def custom_flow(class0, class1, batch_size, augment, SCALE, mask):

    im0 = list(class0)
    im1 = list(class1)
    full_0 = list(class0)
    full_1 = list(class1)
    
    while True:
        iteration = 0
        labels = np.ones(batch_size) * -1
        data = np.zeros((batch_size, 256, 256, 1))
       
    
        while iteration < batch_size:
    
            #Check if all entries used. If TRUE reset table.
            if len(im0) == 0:
                im0 = full_0
            if len(im1) == 0:
                im1 = full_1
            
            #Choose feature.
            whatFeat = np.random.randint(0,2)#iteration % 2
            
            #Add label.
            labels[iteration] = 1
            if (whatFeat == 0):
                labels[iteration] = 0
        
            if whatFeat == 1:
                imagesIn = im1
            
                #Choose random image.
                whatIm = random.randint(0, len(imagesIn) - 1)
                
                #Read in image.
                chosenIm = get_image(imagesIn[whatIm], SCALE, mask)
            
                #Remove chosen from batch.
                im1 = [x for i,x in enumerate(imagesIn) if i!= whatIm] 
                
                #Augment image
                if augment == True:
                    augIm = aug(chosenIm)
                else:
                    augIm = chosenIm

                #Add to dataset.
                data[iteration, :, :, :] = augIm

                #Next iteration.
                iteration += 1
               
            else:
                imagesIn = im0
                
                #Choose random image.
                whatIm = random.randint(0, len(imagesIn) - 1)
                
                #Read in image.
                chosenIm = get_image(imagesIn[whatIm], SCALE, mask)
        
                #Remove chosen from batch.
                im0 = [x for i,x in enumerate(imagesIn) if i!= whatIm] 

                #Augment image
                if augment == True:
                    augIm = aug(chosenIm)
                else:
                    augIm = chosenIm

                #Add to dataset.
                data[iteration, :, :, :] = augIm

                #Next iteration.
                iteration += 1 
            
        #Shuffle.
        perm_im = np.random.permutation(data.shape[0])
        data = data[perm_im]
        labels = labels[perm_im]
        
        yield (data, labels)

def CNN():
    #Common variables
    conv_var = dict(kernel_size = (3,3), activation = 'relu')
    pool_var = dict(pool_size = (2,2))
    
    i = Input(shape = (256,256,1))
    
    x = Conv2D(32, **conv_var)(i)
    x = MaxPooling2D(**pool_var)(x)
    
    x = Conv2D(48, **conv_var)(x)
    x = MaxPooling2D(**pool_var)(x)
    
    x = Conv2D(64, **conv_var)(x)
    x = MaxPooling2D(**pool_var)(x)
    x = Flatten()(x)
    
    x = Dense(64, activation = 'relu')(x)
    x = Dropout(0.5)(x)
    o = Dense(1, activation = 'sigmoid')(x)
    
    model = Model(i, o, name='CNN')
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def get_data(whatDATA):
    #Import meta table.
    tablePATH = storagePATH + '/data/Tables/meta_p95_with_3sig_asym.csv'
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

def test_data(whatDATA, SCALE, mask):
    #Import meta table.
    tablePATH = storagePATH + '/data/Tables/meta_p95_with_3sig_asym.csv'
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

    #Load in images.
    for i in range (len(noFeat_name)):
        imagePATH = storagePATH + '/data/fits_512px/' + noFeat_name[i]
        im = fits.open(imagePATH)
        if mask == True:
            imCropped = cropImage(im[0].data, 256)
        else:
            imCropped = cropImage(im[0].data[0], 256)
        noFeatures.append(imCropped[:,:,np.newaxis])
        target0.append(0)
        im.close()

    for i in range (len(withFeat_name)):
        imagePATH = storagePATH + '/data/fits_512px/' + withFeat_name[i]
        im = fits.open(imagePATH)
        if mask == True:
            imCropped = cropImage(im[0].data, 256)
        else:
            imCropped = cropImage(im[0].data[0], 256)
        withFeatures.append(imCropped[:,:,np.newaxis])
        target1.append(1)
        im.close()

    #Concatenate ALL data.
    if SCALE == True:
        withFeatures = np.array(withFeatures)
        c1 = 255.0/np.log(1 + withFeatures.max())
        noFeatures = np.array(noFeatures)
        c2 = 255.0/np.log(1 + noFeatures.max())

        data = np.concatenate((c1*np.log(1 + withFeatures), c2*np.log(1 + noFeatures)), axis = 0)
    else:
        data = np.concatenate((withFeatures, noFeatures), axis = 0)

    target = np.concatenate((target1, target0), axis = 0)

    #Add colour channel.
    #data = data[:,:,:,np.newaxis]
    
    #return IMAGES
    return np.array(withFeatures), np.array(noFeatures)
