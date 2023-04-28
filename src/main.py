from read_utils import read_numpy_mnist_data, img_2_event_img, read_from_csv
from train import LIFSNN
from training_rules import hebbian_oja
from inference import test_snn_with_ck

import numpy as np

if __name__ == '__main__':
    #image_list, label_list = read_from_csv("./data/ckfiltered.csv")
    image_list, label_list = read_numpy_mnist_data('data/archive-2', 4)
    event_images = []
    for image in image_list:
        event_image = img_2_event_img(image_list, 20)
        event_images.append(event_image)
    input_dim = 48 * 48
    output_dim = 2
    vdecay = 0.3
    vth = 0.2
    snn_timestep = 20
    input_2_output_weight = np.random.rand(output_dim, input_dim)
    network = LIFSNN(input_2_output_weight, input_dim, output_dim, vdecay, vth, snn_timestep)
    train_data = np.array(event_images)
    hebbian_oja(network, train_data, lr=2e-2, epochs=10)
    # test_snn_with_ck()