import numpy as np 
import pandas as pd 

label_number_size = 10

data_batch_1 = 'datasets/cifar-10-batches-py/data_batch_1'
data_batch_2 = 'datasets/cifar-10-batches-py/data_batch_2'
batches_meta = 'datasets/cifar-10-batches-py/batches.meta'
test_batch = 'datasets/cifar-10-batches-py/test_batch'

# pickling batch files
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

"""
Function for loading training, validation and test data
Training data is batch 1
Validation data is batch 2
Test data is test batch
"""
def load_batch(dataset):
    data = unpickle(dataset)
    X = np.transpose(data[b'data']).astype(float)
    y = np.array(data[b'labels']).reshape(-1, 1)
    Y = np.zeros((label_number_size, int(X.shape[1])))
    Y = ((np.arange(y.max()+1) == y[:,:]).astype(int)).T
    return [X, Y, y]

# passed in argument is np array
def preprocess(data):
    mean_data = np.transpose(data.mean(axis=1))
    std_data = np.transpose(data.std(axis=1))
    std_data_newaxis = std_data[:, np.newaxis]
    mean_data_newaxis = mean_data[:, np.newaxis]
    meanX = np.tile(mean_data_newaxis, (1, data.shape[1]))
    stdX = np.tile(std_data_newaxis, (1, data.shape[1]))

    norm = (data - meanX) / stdX
    return norm

"""
Initialize values of W and b to be Gaussian random variables with
zero mean and std of 0.01.
Dimenions of W: Kxd
Dimensions of b: dx1
"""
def init_vars(dimension):
    np.random.seed(500)
    W = np.random.normal(0, 0.01, (label_number_size, dimension))
    b = np.random.normal(0, 0.01, (dimension, 1))
    return [W, b]

if __name__ == "__main__":
    trainX, trainY, trainy = load_batch(data_batch_1)
    valX, valY, valy = load_batch(data_batch_2)
    testX, testY, testy = load_batch(test_batch)
    

