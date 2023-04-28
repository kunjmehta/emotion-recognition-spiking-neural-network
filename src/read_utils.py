import numpy as np
import pandas as pd

def read_from_csv(f):
    df = pd.read_csv(f)
    labels = df[df['Usage'] == 'Training']['emotion'].tolist()
    pixels = df[df['Usage'] == 'Training']['pixels']
    #print(pixels)
    pixels_list = [[int(a) for a in i.split(" ")] for i in pixels]
    print(len(pixels_list), len(pixels_list[0]))
    #pixels = pixels.to_numpy()
    print(df)
    return pixels, labels

def read_numpy_mnist_data(save_root, num_sample):
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
    image_list = np.zeros((num_sample, 640, 490))
    label_list = [0, 0, 1, 1]
    # for ii in range(num_sample):
    #     image_label = pickle.load(open(save_root + '/' + str(ii) + '.p', 'rb'))
    #     image_list[ii] = image_label[0]
    #     label_list.append(image_label[1])

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
    batch_size = 32
    image_size = 48 * 48
    #image = image.reshape(batch_size, image_size, image_size, 1)
    
    #Generate a random image of the shape batch_size x image_size x image_size x snn_timestep. Numpy random rand function will be useful here. 
    random_image = np.random.rand(batch_size, image_size, image_size, snn_timestep)

    #Generate the event image
    event_image = np.where(image >= random_image, 1, 0)
    
    return event_image