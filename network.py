import random
import numpy as np

def sigmoid(z):
		return 1.0/(1.0+np.exp(-z))

class Network(object):

	def __init__(self, sizes):
		self.num_layers = len(sizes)
		self.sizes = sizes
		self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
		self.weights = [np.random.randn(y, x) for x, y, in zip(sizes[:-1], sizes[1:])]

	def feedForward(self, a):
		for b, w in zip(self.biases, self.weight):
			a = sigmoid(np.dot(w, a) + b)
		return a

	def SGD(self, trainign_data, epochs, mini_batch_size, eta, test_data=None):
		n = len(trainign_data)
		for j in range(epochs):
			random.shuffle(trainign_data)
			mini_batches = [
				training_data[k:k+mini_batch_size]
				for k in xrange(0, n, mini_batches)]
			for mini_batches in mini_batches:
				self.update_mini_batch(mini_batch, eta)
			if test_data:
				n_test = len(test_data)
				print ("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
			else:
				print ("Epoch {0} complete".format(j))

	def update_mini_batch(self, mini_batch, eta):
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		for x, y in mini_batch:
			delta_nabla_b, delta_nabla_w = self.backprop(x, y)
			nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
			nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delata_nabla_w)]
		self.weight = [w - (eta/len(mini_batch))*nw
			for w, nw in zip(self.weights, nabla_w)]
		self.biases = [b - (eta/len(mini_batch))*nb
			for b, nb in zip(self.biases, nabla_b)]

	def backprop(self, x, y):
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w  in self.weights]
		activation = x
		activations = [x]
		zs = []
		for b, w in zip(self.biases, self.weights):
			z = np.dot(w, activation) + b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)
		delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
		nabla_b[-1] = delta
		nabla_w[-1] = np.dot(delta, activations[-2].transpose())
		for lastlayer in xrange(2, self.num_layers):
			z = zs[-lastlayer]
			sp = sigmoid_prime(z)
			delta = np.dot(self.weights[-lastlayer + 1].transpose(), delta) *sp
			nabla_b = delta
			nabla_w[-lastlayer] = np.dot(delta, activations[-lastlayer - 1].transpose())

		return (nabla_b, nabla_w)

	def evaluate(self, test_data):
		test_results = [(np.argmax(self.feedForward(x)), y)
			for (x, y) in test_data]
		return sum(int(x==y) for x, y in test_data)

	def cost_derivate(self, output_activations, y):
		return (output_activations - y)



def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
	return sigmoid(z)*sigmoid(1 - sigmoid(z))