from src.read_utils import read_numpy_fer_data, img_2_event_img
from src.network import SNN
from src.training_rules import hebbian_oja, STDP
from src.inference import test_snn_with_fer, test_snn_with_fer_one_instance

import numpy as np


if __name__ == '__main__':
    image_size = 48
    input_dim = image_size * image_size
    output_dim = 2
    vdecay = 0.3
    vth = 0.2
    snn_timestep = 20

    image_list, label_list = read_numpy_fer_data('data/train/happy', num_sample=32, image_size=image_size)
    event_image = img_2_event_img(image_list, snn_timestep)

    # generate random weights
    input_2_output_weight = np.random.rand(output_dim, input_dim)
    network = SNN(input_2_output_weight, input_dim, output_dim, vdecay, vth, snn_timestep)

    # test without learning
    img = event_image[0, :, :, :]
    predicted_output, predicted_class = test_snn_with_fer_one_instance(network, img)

    # test with oja
    hebbian_oja(network, event_image, lr=2e-2, epochs=100)
    predicted_output, predicted_class = test_snn_with_fer_one_instance(network, img)
