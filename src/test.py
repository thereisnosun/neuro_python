import mnist_loader
import network



# training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# net = network.Network([784, 30, 10])
# net.SGD(training_data, 30, 10, 3.0, test_data = test_data)


import network2
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net2 = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
net2.large_weight_initializer()
net2.SGD(training_data, 30, 10, 0.5, evaluation_data=test_data, monitor_evaluation_accuracy=True)

# import network
# from mnist_load import MnistLoad
# loader = MnistLoad('../data-new/')
# train_data, test_data = loader.loadAsNumpyData()
# net = network.Network([784, 30, 10])
# net.SGD(train_data, 30, 10, 3.0, test_data = test_data)