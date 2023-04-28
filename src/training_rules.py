import numpy as np

def hebbian_oja(network, train_data, lr=1e-5, epochs=10):
    """ 
    Function to train a network using Hebbian learning rule
        Args:
            network (SNN): SNN network object
            train_data (list): training data 
            lr (float): learning rate
            epochs (int): number of epochs to train with. Each epoch is defined as one pass over all training samples. 
        
        Write the operations required to compute the weight increment according to the hebbian learning rule. Then increment the network weights. 
    """
    
    #iterate over the epochs
    for _ in range(epochs):
        weight_increment = np.zeros((2, 2304))
        #iterate over all samples in train_data
        for data in train_data:
            
            #compute the firing rate for the input
            r1 = np.sum(data[0][0])/len(data[0][0])
            r2 = np.sum(data[0][1])/len(data[0][1])

            #compute the firing rate for the output
            ro = np.sum(data[1])/len(data[1])

            #compute the correlation using the firing rates calculated above
            correlation1 = r1 * ro
            correlation2 = r2 * ro
            
            oja_term1 = network.input_2_output_connection.weights[0][0] * ro * ro
            oja_term2 = network.input_2_output_connection.weights[0][1] * ro * ro

            #compute the weight increment
            weight_increment[0][0] = lr * (correlation1 - oja_term1)
            weight_increment[0][1] = lr * (correlation2 - oja_term2)
            
            #increment the weight
            network.input_2_output_connection.weights += weight_increment

class STDP():
    """Train a network using STDP learning rule"""
    def __init__(self, network, A_plus, A_minus, tau_plus, tau_minus, lr, snn_timesteps=20, epochs=30, w_min=0, w_max=1):
        """
        Args:
            network (SNN): network which needs to be trained
            A_plus (float): STDP hyperparameter
            A_minus (float): STDP hyperparameter
            tau_plus (float): STDP hyperparameter
            tau_minus (float): STDP hyperparameter
            lr (float): learning rate
            snn_timesteps (int): SNN simulation timesteps
            epochs (int): number of epochs to train with. Each epoch is defined as one pass over all training samples.  
            w_min (float): lower bound for the weights
            w_max (float): upper bound for the weights
        """
        self.network = network
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.snn_timesteps = snn_timesteps
        self.lr = lr
        self.time = np.arange(0, self.snn_timesteps, 1)
        self.sliding_window = np.arange(-4, 4, 1) #defines a sliding window for STDP operation. 
        self.epochs = epochs
        self.w_min = w_min
        self.w_max = w_max
    
    def update_weights(self, t, i):
        """
        Function to update the network weights using STDP learning rule
        
        Args:
            t (int): time difference between postsynaptic spike and a presynaptic spike in a sliding window
            i(int): index of the presynaptic neuron
        
        Fill the details of STDP implementation
        """
        #compute delta_w for positive time difference
        if t>0:
            delta_w = self.A_plus * np.exp(-t / self.tau_plus)

        #compute delta_w for negative time difference
        else:
            delta_w = -self.A_minus * np.exp(-t / self.tau_minus)

        #update the network weights if weight increment is negative
        if delta_w < 0:
            update = self.lr * delta_w * (self.network.input_2_output_connection.weights - self.w_min)
            self.network.input_2_output_connection.weights += update 

        #update the network weights if weight increment is positive
        elif delta_w > 0:
            update = self.lr * delta_w * (self.w_max - self.network.input_2_output_connection.weights)
            self.network.input_2_output_connection.weights += update 

            
    def train_step(self, train_data_sample):
        """
        Function to train the network for one training sample using the update function defined above. 
        
        Args:
            train_data_sample (list): a sample from the training data
            
        This function is complete. You do not need to do anything here. 
        """
        input = train_data_sample[0]
        output = train_data_sample[1]
        for t in self.time:
            if output[t] == 1:
                for i in range(2):
                    for t1 in self.sliding_window:
                        if (0<= t + t1 < self.snn_timesteps) and (t1!=0) and (input[i][t+t1] == 1):
                            self.update_weights(t1, i)
    
    def train(self, training_data):
        """
        Function to train the network
        
        Args:
            training_data (list): training data
        
        This function is complete. You do not need to do anything here. 
        """
        for ee in range(self.epochs):
            for train_data_sample in training_data:
                self.train_step(train_data_sample)

