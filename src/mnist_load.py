import os
import struct
from array import array

import numpy as np

def vectorized_result(j):
	e = np.zeros((10, 1))
	e[j] = 1.0
	return e

class MnistLoad:

	def __init__(self, path):
		self.m_train_image_path = os.path.join(path, 'train-images.idx3-ubyte')
		self.m_train_label_path = os.path.join(path, 'train-labels.idx1-ubyte')

		self.m_test_image_path = os.path.join(path, 't10k-images.idx3-ubyte')
		self.m_test_label_path = os.path.join(path, 't10k-labels.idx1-ubyte')

		self.path = path
		self.train_imgs = []
		self.train_lbls = []
		self.test_imgs = []
		self.test_lbls = []


	def loadAsNumpyData(self):
		self.loadTrainData()
		self.loadTestData()

		train_data_in = [np.reshape(x, (784, 1)) for x in self.train_imgs]
		train_data_out = [vectorized_result(y) for y in self.train_lbls]
		train_data = zip(train_data_in, train_data_out)

		test_data_in = [np.reshape(x, (784, 1)) for x in self.test_imgs]
		test_data = zip(test_data_in, self.test_lbls)

		# train_data_in = [np.reshape(x, (784, 1)) for x in self.train_imgs]
		# train_data_out = [vectorized_result(y) for y in self.train_lbls]
		# train_data = zip(train_data_in, train_data_out)

		# test_data_in = [np.reshape(x, (784, 1)) for x in self.test_imgs]
		# test_data = zip(test_data_in, self.test_lbls)

		# print(type(train_data_in), type(train_data_out), type(train_data))
		# print(len(test_data_in), len(self.test_lbls), len(list(test_data)))

		return train_data, test_data


	def loadTrainData(self):
		self.train_imgs, self.train_lbls = self.load(self.m_train_image_path, self.m_train_label_path)

	def loadTestData(self):
		self.test_imgs, self.test_lbls = self.load(self.m_test_image_path, self.m_test_label_path)

	@classmethod
	def load(cls, path_img, path_lbl):
		with open(path_lbl, 'rb') as file:
			magic, size = struct.unpack(">II", file.read(8))
			if magic != 2049:
				raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))

			labels = array("B", file.read())

		with open(path_img, 'rb') as file:
			magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
			if magic != 2051:
				raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))

			image_data = array("B", file.read())
			print("Rock", len(image_data))

		images = []
		for i in range(size):
			images.append([0] * rows * cols)

		for i in range(size):
			images[i][:] = image_data[i * rows * cols:(i + 1) * rows * cols]

		print("Rock", len(images))
		
		return images, labels
