# -*- coding: utf-8 -*-
"""
NBDNet class and functions 

@author: LiHongxiang
"""

import numpy as np
import tensorflow as tf
from keras.layers import Input, Conv2D, BatchNormalization, Activation
from keras.models import Model
    
class NBDNet(Model):  
  '''
  NBDNet model definition
  '''    
  def __init__(self):
    super().__init__()
    # parameters
    self.image_channels = 8
    self.filters = 128
    self.depth = 13
    # DnCNN
    self.dncnn = tf.keras.Sequential()
    layer_count = 0
    self.dncnn.add(Input(shape=(None, None, self.image_channels), name='input' + str(layer_count)))
    layer_count += 1
    self.dncnn.add(Conv2D(filters=self.filters, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='Orthogonal', padding='same',
               name='conv' + str(layer_count)))
    for i in range(self.depth-2):
        layer_count += 1
        self.dncnn.add(Conv2D(filters=self.filters, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='Orthogonal', padding='same',
                   use_bias=False, name='conv' + str(layer_count)))
        layer_count += 1
        self.dncnn.add(BatchNormalization(axis=3, momentum=0.9, epsilon=0.0001, name='bn' + str(layer_count)))
        layer_count += 1
        self.dncnn.add(Activation('relu', name='relu' + str(layer_count)))
    layer_count += 1    
    self.dncnn.add(Conv2D(filters=self.image_channels, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='Orthogonal',
               padding='same', use_bias=False, name='conv' + str(layer_count))) 

  def call(self, x, training=False): 
      x = tf.nn.space_to_depth(x, 2)
      x = x - self.dncnn(x)
      x = tf.nn.depth_to_space(x, 2)
      return x 

def multi_array(array, look_y, look_x):
    '''
    divides array into non-overlapping rectangular blocks of size look_y × look_x,
    and then computes the average values of each block.
    This operation reduces the size of the array by a factor of look_y and look_x.
        
    Parameters
    ----------
    array (2D numpy.ndarray): the array to be multi-looked.
    look_y (int): the multi_looking factor along the y axis.
    look_x (int): the multi_looking factor along the x axis.
    
    Returns
    -------
    array_multi (2D numpy.ndarray): the multi-looked array.
    '''  
    height, width = array.shape
    height_multi, width_multi = height // look_y, width // look_x
    array_multi = np.zeros((height_multi, width_multi), dtype = array.dtype)
    for yy in range(look_y):
        for xx in range(look_x):
            array_multi += array[yy:yy+height_multi*look_y:look_y,xx:xx+width_multi*look_x:look_x]
    array_multi /= look_y * look_x
    return array_multi

def slc_intf(slc1, slc2, look_y, look_x, method = 0):
    '''
    Multi-look interferogram generation from co-registered single look complex (SLC) images.
    The multi-looking factors are look_y and look_x. 
    
    Parameters
    ----------
    slc1 (2D numpy.ndarray): the master SLC.
    slc2 (2D numpy.ndarray): the slave SLC.
    look_y (int): the multi_looking factor along the y axis.
    look_x (int): the multi_looking factor along the x axis.
    method (int): 0 when the true amplitudes of the two SLCs are equal (default), 1 for general case
    
    Returns
    -------
    denoised (2D numpy.ndarray): the denoised interferogram, 
    whose magnitude and phase correspond to the denoised coherence and phase, respectively.
    '''
    mli1 = multi_array(np.abs(slc1)**2, look_y, look_x)
    mli2 = multi_array(np.abs(slc2)**2, look_y, look_x)
    intf = multi_array(slc1*np.conj(slc2), look_y, look_x)
    if method == 0:
        intf_norm = intf/(mli1+mli2)*2 
    else:
        intf_norm = intf/((mli1*mli2)**(1/2)) 
    return intf_norm

def denoise_multi(noisy_intf, model_path = 'NBDNet_model.hdf5'):
    '''
    NBDNet denoising for multi-look interferogram. 
    
    Parameters
    ----------
    noisy_intf (2D numpy.ndarray): the multi-look interferogram.
    model_path (str): the filepath of NBDNet_model.hdf5.
    
    Returns
    -------
    denoised (2D numpy.ndarray): the denoised multi-look interferogram, 
    whose magnitude and phase correspond to the denoised coherence and phase, respectively.
    '''
    # load model
    model = NBDNet()
    model.build(input_shape=(None,None,None,2))
    model.load_weights(model_path)
    #print(model.summary())
    # adjust input
    height, width = noisy_intf.shape
    noisy_intf = np.pad(noisy_intf,((0,height % 2),(0,width % 2)),'reflect')
    noisy_intf = np.stack((np.real(noisy_intf), np.imag(noisy_intf)), 2)[np.newaxis,...]
    # denoising
    denoised = model(noisy_intf).numpy() 
    denoised = denoised[0, :, :, 0] + 1j * denoised[0, :, :, 1]
    denoised = denoised[0:height,0:width] 
    return denoised

def denoise_single(slc1,slc2, model_path = 'NBDNet_model.hdf5'):
    '''
    NBDNet denoising for single-look interferogram. 
    
    Parameters
    ----------
    slc1 (2D numpy.ndarray): the master SLC.
    slc2 (2D numpy.ndarray): the slave SLC.
    model_path (str): the filepath of NBDNet_model.hdf5.
    
    Returns
    -------
    denoised (2D numpy.ndarray): the denoised single-look interferogram, 
    whose magnitude and phase correspond to the denoised coherence and phase, respectively.
    '''
    # load model
    model = NBDNet()
    model.build(input_shape=(None,None,None,2))
    model.load_weights(model_path)
    #print(model.summary())
    # adjust input
    def cal_pad_size(N):
        if N % 4 == 1:
            return 2
        elif N % 4 == 2:
            return 1
        elif N % 4 == 3:
            return 0
        else:
            return 3
    height, width = slc1.shape
    slc1 = np.pad(slc1,((1,1 + cal_pad_size(height)),(1,1 + cal_pad_size(width))),'reflect')
    slc2 = np.pad(slc2,((1,1 + cal_pad_size(height)),(1,1 + cal_pad_size(width))),'reflect')
    height_pad, width_pad = slc1.shape
    noisy_intf = np.zeros((4,(height_pad-1)//2, (width_pad-1)//2), dtype = 'complex64')
    noisy_intf[0,:,:] = slc_intf(slc1[0:height_pad-1,0:width_pad-1], slc2[0:height_pad-1,0:width_pad-1], 2, 2)
    noisy_intf[1,:,:] = slc_intf(slc1[0:height_pad-1,1:width_pad], slc2[0:height_pad-1,1:width_pad], 2, 2)
    noisy_intf[2,:,:] = slc_intf(slc1[1:height_pad,0:width_pad-1], slc2[1:height_pad,0:width_pad-1], 2, 2)
    noisy_intf[3,:,:] = slc_intf(slc1[1:height_pad,1:width_pad], slc2[1:height_pad,1:width_pad], 2, 2)  
    # denoising
    noisy_intf = np.stack((np.real(noisy_intf), np.imag(noisy_intf)), 3)
    denoised_cube = model(noisy_intf).numpy() 
    denoised_cube = denoised_cube[:, :, :, 0] + 1j * denoised_cube[:, :, :, 1]
    # aggregation
    denoised = np.zeros((height_pad-2, width_pad-2), dtype = 'complex64')
    denoised[0::2,0::2] = np.mean(denoised_cube, axis = 0)
    denoised[0::2,1::2] = (denoised_cube[0,:,1:] + denoised_cube[1,:,:-1] + denoised_cube[2,:,1:] + denoised_cube[3,:,:-1])/4;
    denoised[1::2,0::2] = (denoised_cube[0,1:,:] + denoised_cube[1,1:,:] + denoised_cube[2,:-1,:] + denoised_cube[3,:-1,:])/4;
    denoised[1::2,1::2] = (denoised_cube[0,1:,1:] + denoised_cube[1,1:,:-1] + denoised_cube[2,:-1,1:] + denoised_cube[3,:-1,:-1])/4;  
    denoised = denoised[0:height,0:width] 
    return denoised


