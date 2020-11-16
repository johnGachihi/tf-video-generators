import os

import pandasgeneratorutils
from predictiongenerator import PredictionGenerator

import pandas as pd
from pathlib import Path


class PandasPredictionGenerator(PredictionGenerator):
    """
    :parameter transformation
    """
    def __init__(self, source: pd.DataFrame, data_path: Path, **kwargs):
        """
        :param source: A pd.DataFrame with a list of samples
        :key batch_size: default: 10. Batch size.
        :key nb_frames: default: 10. Number of frames in sample,
        :key printer: default: `print`. Function for printing sample details,
        :key frame_size: default: (224, 224).
        """
        self.data_path = data_path
        super().__init__(source, **kwargs)

    def assert_source_validity(self, source):
        pass

    def get_sample_list(self, source: pd.DataFrame, printer=print) -> list:
        samples = []
        for [sample] in source.values:
            sample_path = self.data_path / str(sample)
            if len(os.listdir(sample_path)) >= self.nb_frames:
                samples.append(sample_path)

        printer(f'Sample size: {len(samples)}')

        return samples

    def process_sample(self, sample):
        return pandasgeneratorutils.process_sample(
            sample, self.nb_frames, self.frame_size)