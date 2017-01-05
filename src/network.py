"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

class Network(object):
    def __init__(self, sizes, weights_filename = None, biases_filename = None, mask_threshold = 0):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        if biases_filename:
            self.load_biases(biases_filename)
        else:
            self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        if weights_filename:
            self.load_weights(weights_filename)
        else:
            self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

        self.mask = [np.ones((y, x)) for x, y in zip(sizes[:-1], sizes[1:])]
        self.mask_threshold = mask_threshold

    def set_mask_threshold(self, mask_threshold):
        self.mask_threshold = mask_threshold

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def update_mask(self):
        for i in range(self.num_layers - 1):
            self.mask[i] = self.mask[i] * (np.absolute(self.weights[i]) > self.mask_threshold)

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data = None, pruning = True):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data:
            print "Before training: {0} / {1}".format(
                self.evaluate(test_data), len(test_data))

        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, pruning)
            if test_data:
               print "Epoch {0}/ Before Mask: {1} / {2}".format(
                    j, self.evaluate(test_data), len(test_data))
            if pruning:
                self.update_mask()
                self.weights = [w*m for w,m in zip(self.weights, self.mask)]
                if test_data:
                    print "Epoch {0}/ After Mask: {1} / {2}".format(
                        j, self.evaluate(test_data), len(test_data))

    def update_mini_batch(self, mini_batch, eta, pruning = True):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        (grad_w, grad_b) = self.get_gradients(mini_batch, eta, pruning)
        self.weights = [w - gw
                        for w, gw in zip(self.weights, grad_w)]
        self.biases = [b - gb
                       for b, gb in zip(self.biases, grad_b)]

    def get_gradients(self, mini_batch, eta, pruning = True):
        """Gets network's weight and bias gradients by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate.
        Returns a tuple (weights_gradients, bias_gradients)"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        grad_w = [nw*eta/len(mini_batch)
                        for nw in nabla_w]
        grad_b = [nb*eta/len(mini_batch)
                       for nb in nabla_b]
#        if pruning:
        grad_w = [g*m for g,m in zip(grad_w, self.mask)]
        return (grad_w, grad_b)

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

    def save_weights(self, weights_filename):
        np.save(weights_filename, self.weights)

    def load_weights(self, weights_filename):
        self.weights = np.load(weights_filename)

    def save_biases(self, biases_filename):
        np.save(biases_filename, self.biases)

    def load_biases(self, biases_filename):
        self.biases = np.load(biases_filename)

    def save(self, weights_filename, biases_filename):
        self.save_weights(weights_filename)
        self.save_biases(biases_filename)

    def load(self, weights_filename, biases_filename):
        self.load_weights(weights_filename)
        self.load_biases(biases_filename)

    def weights_count(self, layer_index):
        return self.weights[layer_index].size

    def total_weights_count(self):
        return sum([x*y for x, y in zip(self.sizes[:-1], self.sizes[1:])])

    def weights_under(self):
        total = 0
        for i in range(self.num_layers - 1):
            total +=  (np.absolute(self.weights[i]) < self.mask_threshold).sum()
        return total/float(self.total_weights_count())
        
#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
