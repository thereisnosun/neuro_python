import mnist_loader
import network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data = test_data)


# import network
# from mnist_load import MnistLoad
# loader = MnistLoad('../data-new/')
# train_data, test_data = loader.loadAsNumpyData()
# net = network.Network([784, 30, 10])
# net.SGD(training_data, 30, 10, 3.0, test_data = test_data)