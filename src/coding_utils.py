import numpy as np
from scipy.stats import poisson, bernoulli

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

#################################################################################################################################

def encode_bernoulli(image, time_bin, spike_prob):
    # Compute the number of time bins
    num_time_bins = int(np.ceil(image.size / spike_prob))

    # Reshape the image into a 1D array
    image = image.ravel()

    # Initialize the spike train
    spike_train = np.zeros((num_time_bins, image.size), dtype=np.uint8)

    # Generate spikes using Bernoulli coding
    for i in range(num_time_bins):
        # Generate random Bernoulli spikes for each pixel
        spikes = bernoulli.rvs(spike_prob, size=image.size)

        # Set the pixel value based on the Bernoulli spikes
        pixel_values = spikes * image

        # Add the pixel values to the spike train
        spike_train[i] = pixel_values

    return spike_train.astype(np.uint8)

def encode_bernoulli(image, spike_prob):
    # Compute the number of time bins
    num_time_bins = int(np.ceil(image.size / spike_prob))

    # Reshape the image into a 1D array
    image = image.ravel()

    # Initialize the spike train
    spike_train = np.zeros((num_time_bins, image.size), dtype=np.uint8)

    # Generate spikes using Bernoulli coding
    for i in range(num_time_bins):
        # Generate random Bernoulli spikes for each pixel
        spikes = bernoulli.rvs(spike_prob, size=image.size)

        # Set the pixel value based on the Bernoulli spikes
        pixel_values = np.where(spikes == 1, image, 0)

        # Add the pixel values to the spike train
        spike_train[i] = pixel_values

    return spike_train.astype(np.uint8)


def decode_bernoulli(spike_train):
    # Compute the number of time bins
    num_time_bins = spike_train.shape[0]

    # Reshape the spike train into a 2D array
    spike_train = spike_train.reshape(num_time_bins, -1)

    # Initialize the decoded image
    decoded_image = np.zeros(spike_train.shape[1], dtype=np.uint8)

    # Decode the spike train using inverse Bernoulli coding
    for i in range(num_time_bins):
        # Find the indices of pixels that fired spikes
        indices = np.nonzero(spike_train[i])[0]

        # Set the pixel values to the mean of the Bernoulli distribution
        decoded_image[indices] = int(np.round(np.mean(spike_train[i, indices])))

    # Reshape the decoded image to the original shape
    decoded_image = decoded_image.reshape(-1, spike_train.shape[1] // decoded_image.shape[0])

    return decoded_image


def encode_temporal(image, threshold=128, duration=10):
    # Convert the image to a 1D array
    image = image.flatten()

    # Initialize the spike train
    spike_train = np.zeros((len(image), duration), dtype=int)

    # Encode the image using temporal coding
    for i in range(len(image)):
        if image[i] > threshold:
            spike_train[i, :] = 1

    return spike_train

def decode_temporal(spike_train):
    # Compute the number of neurons and time bins
    num_neurons, _ = spike_train.shape

    # Initialize the decoded image
    decoded_image = np.zeros(num_neurons, dtype=np.uint8)

    # Decode the spike train using temporal decoding
    for i in range(num_neurons):
        # Compute the indices of time bins where the neuron spiked
        indices = np.where(spike_train[i, :] > 0)[0]

        # Set the pixel value to the mean of the spike times
        if len(indices) > 0:
            decoded_image[i] = int(np.round(np.mean(indices)))

    # Reshape the decoded image to the original shape
    decoded_image = decoded_image.reshape(-1, spike_train.shape[0])

    return decoded_image

##################################################################################################################################

def encode_population(image, num_neurons=100, threshold=128):
    # Convert the image to a 1D array
    image = image.flatten()

    # Initialize the spike train
    spike_train = np.zeros((num_neurons, len(image)), dtype=int)

    # Generate random receptive fields for the neurons
    receptive_fields = np.random.normal(loc=threshold, scale=threshold/2, size=num_neurons)

    # Encode the image using population coding
    for i in range(num_neurons):
        neuron_activity = (image - receptive_fields[i]) / threshold
        spike_train[i, :] = (neuron_activity > 0).astype(int)

    return spike_train

def decode_population(spike_train, threshold=128):
    # Compute the receptive fields for the neurons
    receptive_fields = np.mean(threshold * spike_train, axis=1)

    # Compute the decoded image by summing over the neuron activities
    decoded_image = np.sum(spike_train - receptive_fields[:, np.newaxis], axis=0)

    # Reshape the decoded image to its original shape
    decoded_image = decoded_image.reshape((-1, spike_train.shape[1]))

    return decoded_image

def bernoulli_coding(image, time_steps = 100):
    """
    Encode an input grayscale image into a spike train using Bernoulli coding.
    Args:
        image (numpy.ndarray): A grayscale image represented as a 2D numpy array.
        time_steps (int): The number of time steps to generate spike trains for.
    Returns:
        numpy.ndarray: A spike train represented as a 2D numpy array where each row represents a time step 
        and each column represents a pixel in the original image.
    """
    # Flatten the image
    image = image.flatten()
    # Normalize the pixel values
    image = image / 255.0
    # Generate spike train using Bernoulli distribution
    spike_train = np.zeros((len(image), time_steps))
    for i in range(time_steps):
        spike_train[:, i] = np.random.binomial(1, image)
    return np.reshape(spike_train / np.max(spike_train), (48, 48, -1))

def poisson_coding(image, time_steps = 100):
    """
    Encode an input grayscale image into a spike train using Poisson coding.
    Args:
        image (numpy.ndarray): A grayscale image represented as a 2D numpy array.
        time_steps (int): The number of time steps to generate spike trains for.
    Returns:
        numpy.ndarray: A spike train represented as a 2D numpy array where each row represents a time step 
        and each column represents a pixel in the original image.
    """
    # Flatten the image
    image = image.flatten()
    # Normalize the pixel values
    image = image / 255.0
    # Generate spike train using Poisson distribution
    spike_train = np.zeros((len(image), time_steps))
    for i in range(time_steps):
        spike_train[:, i] = np.random.poisson(image)
    return np.reshape(spike_train / np.max(spike_train), (48, 48, -1))
