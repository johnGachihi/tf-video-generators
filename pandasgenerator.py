from generator import Generator
import pandas as pd
import math
from pathlib import Path


class PandasGenerator(Generator):
    def __init__(self, source, data_path: Path):
        self.data_path = data_path
        super().__init__(source)

    def assert_source_validity(self, source: pd.DataFrame):
        if not isinstance(source, pd.DataFrame):
            raise TypeError('`source` should be a DataFrame')

        if not self.data_path.exists():
            raise FileNotFoundError('`data_path` not found')

    def get_sample_list(self, source: pd.DataFrame):
        return [[self.data_path / str(sample), label]
                for [sample, label] in source.values.tolist()]

    def process_sample(self, sample):
        # Get images in sample

            # Pick when more than nb_frames
            # Return list of Paths
        # Convert images to array
        pass

    @staticmethod
    def get_images(sample: Path, nb_samples: int):
        return [img for img in sample.iterdir()]

    @staticmethod
    def pick_at_intervals(a: list, max: int) -> list:
        if len(a) == max:
            return a
        else:
            step = len(a) / max
            picked = []
            for i in range(1, max+1):
                picked.append(a[math.floor(i * step) - 1])

            return picked

