from neurons import LIFNeurons, Connections
import numpy as np

class LIFSNN:
    """ Define a Spiking Neural Network with No Hidden Layer """

    def __init__(self, input_2_output_weight, 
                 input_dimension=2, output_dimension=2,
                 vdecay=0.5, vth=0.5, snn_timestep=20):
        """
        Args:
            input_2_hidden_weight (ndarray): weights for connection between input and hidden layer
            hidden_2_output_weight (ndarray): weights for connection between hidden and output layer
            input_dimension (int): input dimension
            hidden_dimension (int): hidden_dimension
            output_dimension (int): output_dimension
            vdecay (float): voltage decay of LIF neuron
            vth (float): voltage threshold of LIF neuron
            snn_timestep (int): number of timesteps for inference
        """
        self.snn_timestep = snn_timestep
        self.output_layer = LIFNeurons(output_dimension, vdecay, vth)
        self.input_2_output_connection = Connections(input_2_output_weight, input_dimension, output_dimension)
    
    def __call__(self, spike_encoding):
        """
        Args:
            spike_encoding (ndarray): spike encoding of input
        Return:
            spike outputs of the network
        """
        spike_output = np.zeros(self.output_layer.dimension)
        for tt in range(self.snn_timestep):
            input_2_output_psp = self.input_2_output_connection(spike_encoding[:, tt])
            output_spikes = self.output_layer(input_2_output_psp)
            spike_output += output_spikes
        return spike_output/self.snn_timestep
    