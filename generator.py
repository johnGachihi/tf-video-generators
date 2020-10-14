import math
from abc import abstractmethod
from tensorflow.keras.utils import Sequence
import numpy as np
from typing import Tuple


class Generator(Sequence):
    def __init__(self, source, batch_size=5, nb_frames=10, transformations=None, printer=print):
        self.assert_source_validity(source)

        self.nb_frames = nb_frames
        self.batch_size = batch_size
        self.transformations = transformations
        self.samples = self.get_sample_list(source, printer)

    @abstractmethod
    def assert_source_validity(self, source):
        pass

    @abstractmethod
    def get_sample_list(self, source, printer=print) -> list:
        """
        Retrieve a list of samples with their labels

        :param source: The source or an identifier of the
         source data eg. a dataframe, folder path, etc
        :param printer: Function to be used to print sample
         stats.
        """
        pass

    @abstractmethod
    def process_sample(self, sample):
        """
        Process a sample. To return sample data
        :param sample: Process
        :return:
        """
        pass

    def __len__(self):
        return math.ceil(len(self.samples) / self.batch_size)

    def __getitem__(self, idx) -> Tuple[np.array, np.array]:
        batch_samples = self.samples[idx * self.batch_size :
                                     (idx+1) * self.batch_size]
        X, y = [], []
        for sample, label in batch_samples:
            X.append(self.process_sample(sample))
            y.append(label)

        return np.array(X), np.array(y)
