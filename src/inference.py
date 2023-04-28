import numpy as np

def test_snn_with_fer_one_instance(network, img, img_size=48):
    img = np.reshape(img, (img_size * img_size, -1))
    predicted_output = network(img)
    predicted_class = np.argmax(predicted_output)

    return predicted_output, predicted_class

def test_snn_with_fer(network, data_save_dir, data_sample_num):
    """
    Test SNN with MNIST test data
    Args:
        network (SNN): defined SNN network
        data_save_dir (str): directory for the test data
        data_sample_num (int): number of test data examples
    """
    #Read image and labels using the read function
    test_image_list, test_label_list = read_numpy_mnist_data(data_save_dir, data_sample_num)
    
    #Convert the images to event images
    test_event_image_list = img_2_event_img(test_image_list, network.snn_timestep)
    
    #Initialize number of correct predictions to 0
    correct_prediction = 0
    
    #Loop through the test images
    for idx in range(test_image_list.shape[0]):
        #Compute network output for each image. You might have to reshape the image using Numpy reshape function so that its appropriate for the SNN
        test_event_image = test_event_image_list[idx, :, :, :]
        test_event_image = np.reshape(test_event_image, (28*28, -1))
        predicted_output = network(test_event_image)
                
        #Determine the class of the image from the network output. Numpy argmax function might be useful here
        predicted_class = np.argmax(predicted_output)
    
        #Compare the predicted class against true class and update correct_prediction counter
        if predicted_class == test_label_list[idx]:
            correct_prediction += 1
        
    #Compute test accuracy
    test_accuracy = correct_prediction / test_image_list.shape[0]

    return test_accuracy
