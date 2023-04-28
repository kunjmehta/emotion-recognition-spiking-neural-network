import numpy as np
import pandas as pd
import os
from PIL import Image

from src.coding_utils import bernoulli_coding, population_encoding, poisson_coding

def read_numpy_fer_data(save_root, num_sample, image_size=48, mode='train'):
    """
    Read saved numpy MNIST data
    Args:
        save_root (str): path to the folder where the MNIST data is saved
        num_sample (int): number of samples to read
    Returns:
        image_list: list of MNIST image
        label_list: list of corresponding labels
    
    This function is complete. You do not need to do anything here.
    """
    image_list = np.zeros((num_sample, image_size, image_size))
    file_names = os.listdir('data/{}/sad'.format(mode))[:num_sample]
    label_list = []
    for ii in range(num_sample):
        image_list[ii] = np.asarray(Image.open(save_root + '/' + file_names[ii]))
        label_list.append(1)

    return image_list, label_list

def img_2_event_img(image, snn_timestep):
    """
    Transform image to spikes, also called an event image
    Args:
        image (ndarray): image of shape batch_size x 28 x 28
        snn_timestep (int): spike timestep
    Returns:
        event_image: event image- spike encoding of the image
        
    Complete the expression for converting the image to spikes (event image)
    """
    
    #Reshape the image. Do not touch this code
    batch_size = image.shape[0]
    image_size = image.shape[2]
    image = image.reshape(batch_size, image_size, image_size, 1)
    
    #Generate a random image of the shape batch_size x image_size x image_size x snn_timestep. Numpy random rand function will be useful here. 
    # random_image = np.random.rand(batch_size, image_size, image_size, snn_timestep)
    # event_image = np.where(image >= random_image, 1, 0)

    event_image = np.zeros((batch_size, image_size, image_size, snn_timestep))
    for idx in range(len(image[0])):
        event_image[idx, :, :, :] = poisson_coding(image[idx, :, :, :], snn_timestep)

    return event_image