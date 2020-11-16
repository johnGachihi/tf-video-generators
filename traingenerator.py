from abc import abstractmethod
from typing import Tuple

import numpy as np
import pandas as pd
import math

from generator import Generator


class TrainGenerator(Generator):
    def __init__(self, source: pd.DataFrame, transformations: list, **kwargs):
        super().__init__(source, **kwargs)
        self.transformations = transformations

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
        batch_samples = self.samples[idx * self.batch_size:
                                     (idx + 1) * self.batch_size]
        X, y = [], []
        for sample, label in batch_samples:
            X.append(self.process_sample(sample))
            y.append(label)

        return np.array(X), np.array(y)
