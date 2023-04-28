import numpy as np
from scipy.stats import poisson

def poisson_encoding(image, intensity=1.0, time_step=0.001, refractory_period=0.002):
    """
    Encode an nxn image as a spike train using a Poisson process.

    Parameters
    ----------
    image : ndarray
        The nxn image to be encoded as a spike train.
    intensity : float
        The intensity of the Poisson process.
    time_step : float
        The time step for the spike train.
    refractory_period : float
        The refractory period for the spike train.

    Returns
    -------
    ndarray
        A 2D array containing the spike times for each pixel.
    """
    # Calculate the mean firing rate for each pixel
    mean_rates = intensity * image
    
    # Generate a Poisson spike train for each pixel
    spike_trains = []
    for i in range(image.shape[0]):
        row_spike_train = []
        for j in range(image.shape[1]):
            mean_rate = mean_rates[i,j]
            poisson_spike_train = poisson.rvs(mean_rate*time_step, size=int(np.ceil(image.max())))
            refractory_mask = np.zeros_like(poisson_spike_train)
            refractory_mask[1:] = np.diff(poisson_spike_train) < refractory_period
            poisson_spike_train[refractory_mask] = 0
            row_spike_train.append(poisson_spike_train)
        spike_trains.append(row_spike_train)
    
    return np.array(spike_trains)


def binomial_encoding(image, intensity=1.0, time_step=0.001):
    """
    Encode an nxn image as a spike train using a Bernoulli process.

    Parameters
    ----------
    image : ndarray
        The nxn image to be encoded as a spike train.
    intensity : float
        The intensity of the Bernoulli process.
    time_step : float
        The time step for the spike train.

    Returns
    -------
    ndarray
        A 2D array containing the spike times for each pixel.
    """
    # Calculate the probability of firing for each pixel
    firing_probabilities = intensity * image
    
    # Generate a Bernoulli spike train for each pixel
    spike_trains = []
    for i in range(image.shape[0]):
        row_spike_train = []
        for j in range(image.shape[1]):
            firing_probability = firing_probabilities[i,j]
            bernoulli_spike_train = np.random.binomial(1, firing_probability, size=int(np.ceil(image.max())))
            spike_times = np.nonzero(bernoulli_spike_train)[0] * time_step
            row_spike_train.append(spike_times)
        spike_trains.append(row_spike_train)
    
    return np.array(spike_trains)


def temporal_encoding(image, threshold=0.5, time_step=0.001):
    """
    Encode an nxn image as a spike train using temporal coding.

    Parameters
    ----------
    image : ndarray
        The nxn image to be encoded as a spike train.
    threshold : float
        The threshold for spiking activity.
    time_step : float
        The time step for the spike train.

    Returns
    -------
    ndarray
        A 2D array containing the spike times for each pixel.
    """
    # Create an empty array to store the spike trains
    spike_trains = np.zeros(image.shape, dtype=object)
    
    # Loop over each pixel in the image
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Initialize the spike train for the current pixel
            spike_train = []
            
            # Loop over each time step in the image
            for t in range(image.shape[2]):
                # Check if the pixel exceeds the threshold at the current time step
                if image[i,j,t] > threshold:
                    # If so, add the current time step to the spike train
                    spike_train.append(t * time_step)
            
            # Add the spike train to the output array
            spike_trains[i,j] = spike_train
    
    return spike_trains


def population_encoding(image, num_neurons=100, max_rate=100, time_step=0.001):
    """
    Encode an nxn image as a spike train using population coding.

    Parameters
    ----------
    image : ndarray
        The nxn image to be encoded as a spike train.
    num_neurons : int
        The number of neurons in the population.
    max_rate : float
        The maximum firing rate for a neuron.
    time_step : float
        The time step for the spike train.

    Returns
    -------
    ndarray
        A 2D array containing the spike times for each neuron.
    """
    # Flatten the image into a 1D array
    flat_image = image.flatten()
    
    # Normalize the pixel values to be between 0 and 1
    flat_image /= flat_image.max()
    
    # Calculate the preferred stimuli for each neuron
    pref_stimuli = np.linspace(0, 1, num_neurons)
    
    # Calculate the tuning curve for each neuron
    sigma = 0.1
    tuning_curves = np.exp(-(pref_stimuli[:,np.newaxis] - flat_image)**2 / (2*sigma**2))
    
    # Calculate the firing rate for each neuron
    firing_rates = max_rate * tuning_curves.sum(axis=1)
    
    # Generate a Poisson spike train for each neuron
    spike_trains = []
    for i in range(num_neurons):
        firing_rate = firing_rates[i]
        poisson_spike_train = np.random.poisson(firing_rate * time_step, size=int(np.ceil(image.max())))
        spike_trains.append(poisson_spike_train)
    
    return np.array(spike_trains)


def decode(spikes):
    pass
