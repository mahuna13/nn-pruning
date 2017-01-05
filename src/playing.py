import mnist_loader as ml
import numpy as np
import network

def load_network(sizes, weights_filename, biases_filename):
    return network.Network(sizes, weights_filename, biases_filename)

def evaluate_network(network, data):
    print network.evaluate(data)/float(len(data))

def test_network(network):
    training_data, validation_data, test_data = ml.load_data_wrapper()
    evaluate_network(network, test_data)

def validate_network(network):
    training_data, validation_data, test_data = ml.load_data_wrapper()
    evaluate_network(network, validation_data)

def test_network_on_training_data(network):
    training_data, validation_data, test_data = ml.load_data_wrapper()
    test_data_from_training_data = [(x[0], ml.scalar_result(x[1])) for x in training_data]
    evaluate_network(network, test_data_from_training_data)

def train_network(network):
    training_data, validation_data, test_data = ml.load_data_wrapper()

 #   small_training_data = training_data[:5000]
#    test_data_from_training_data = [(x[0], ml.scalar_result(x[1])) for x in small_training_data]
#    network.SGD(training_data, 5, 10, 3.0, test_data = test_data_from_training_data)
    network.SGD(training_data, 50, 10, 3.0, pruning = False)

def load():
    return load_network([784, 300, 100, 10], "weights-300100.npy", "biases-300100.npy")

def play():
    net = network.Network([784, 300, 100, 10])
    train_network(net)
    net.save("weights-300100", "biases-300100")
    return net
#print "Playing around with neural network package"
#net = network.Network([784, 30, 10])

#save on disk
#net.save("weights-30", "biases-30")
