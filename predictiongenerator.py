import math
from abc import abstractmethod

import numpy as np

from generator import Generator


class PredictionGenerator(Generator):
    @abstractmethod
    def get_sample_list(self, source, printer=print) -> list:
        pass

    @abstractmethod
    def process_sample(self, sample):
        pass

    def __len__(self):
        return math.ceil(len(self.samples) / self.batch_size)

    def __getitem__(self, idx) -> np.array:
        batch_samples = self.samples[idx * self.batch_size :
                                     (idx+1) * self.batch_size]
        X = []
        for sample in batch_samples:
            X.append(self.process_sample(sample))

        return np.array(X)
