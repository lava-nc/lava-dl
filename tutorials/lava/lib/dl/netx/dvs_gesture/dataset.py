# TODO: BSD license

# TODO make it automatically download the data
import numpy as np

class Event():
    """A class to hold events
    """
    def __init__(self, filename):
        
        with open(filename, 'rb') as fileHandle:
            rawData = np.fromfile(fileHandle, dtype=np.uint8).astype('uint')
        
        self.x = rawData[0::5]
        self.y = rawData[1::5]
        self.p = rawData[2::5]>>7
        self.p -= np.min(self.p)
        self.t = ( (rawData[2::5]<<16) | (rawData[3::5]<<8) | (rawData[4::5]) ) & 0x7FFFFF
        self.t = self.t//1000

class IBMGestureDataset:
    """Interface to IBM Gesture Test Dataset
    """
    def __init__(self, path, steps_per_sample, blank_time):
        self.path = path
        self.steps_per_sample = steps_per_sample
        self.blank_time = blank_time

        test_params = np.loadtxt(self.path + '/test.txt').astype('int')
        self.file_numbers = test_params[:, 0]
        self.labels = test_params[:, 1]
        self.subsample = 4
        self.x_dim = 128 // self.subsample
        self.y_dim = 128 // self.subsample
        self.f_dim = 2

    def __getitem__(self, index):
        filename = self.path + '/' + str(self.file_numbers[index]) + '.bs2'
        data = Event(filename)
        label = self.labels[index]
        dense_data = np.zeros((self.x_dim,
                               self.y_dim,
                               self.f_dim,
                               self.steps_per_sample + self.blank_time))

        valid = np.argwhere(data.t < self.steps_per_sample)
        dense_data[data.x[valid] // self.subsample,
                   data.y[valid] // self.subsample,
                   data.p[valid],
                   data.t[valid].astype(int)] = 1

        return dense_data, label

    def __len__(self):
        return len(self.labels)
