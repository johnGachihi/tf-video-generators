import math
from abc import abstractmethod
from tensorflow.keras.utils import Sequence
import numpy as np


class Generator(Sequence):
    def __init__(self, source, batch_size=5):
        self.assert_source_validity(source)

        self.batch_size = batch_size
        self.samples = self.get_sample_list(source)

    @abstractmethod
    def assert_source_validity(self, source):
        pass

    @abstractmethod
    def get_sample_list(self, source) -> list:
        """
        Retrieve a list of samples with their labels

        :param source: The source or an identifier of the
         source data eg. a dataframe, folder path, etc
        """
        pass

    @abstractmethod
    def process_sample(self, sample):
        """
        :param sample: Process
        :return:
        """
        pass

    def __len__(self):
        return math.ceil(len(self.samples) / self.batch_size)

    def __getitem__(self, idx):
        batch_samples = self.samples[idx * self.batch_size :
                                     (idx+1) * self.batch_size]
        X, y = [], []
        for sample, label in batch_samples:
            X.append(self.process_sample(sample))
            y.append(label)

        return np.array(X), np.array(y)
