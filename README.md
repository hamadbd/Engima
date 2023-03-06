# Engima

In this example, we use TensorFlow and Keras to build a neural network that can classify images from the CIFAR-10 dataset, which consists of 50,000 training images and 10,000 test images of 32x32 pixel size.

We start by loading the data and preprocessing it by scaling the pixel values to a range between 0 and 1, and converting the labels to one-hot encoded vectors.

Next, we define the neural network architecture, which consists of three convolutional layers with max pooling, followed by a fully connected layer and a softmax output layer.

We then compile the model using the Adam optimizer and categorical cross-entropy loss, and train it on the training set for 10 epochs with a batch size of 32.

Finally, we evaluate the performance of the model on the test set by computing the loss and accuracy, which are printed to the console.

This example demonstrates how Enigma AI can be used to build sophisticated models that can learn complex patterns in large datasets and make accurate predictions.
