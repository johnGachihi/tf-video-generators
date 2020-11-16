import os
from pathlib import Path

import pandas as pd
import numpy as np

import pandasgeneratorutils
from generatorutils import GeneratorUtils
from traingenerator import TrainGenerator


class PandasGenerator(TrainGenerator):
    def __init__(
            self,
            source: pd.DataFrame,
            data_path: Path,
            batch_size=5,
            nb_frames=8,
            transformations=None,
            labelling_strategy='categorical',
            printer=print,
            frame_size=(224, 224),
            dtype=np.uint8
    ):
        self.data_path = data_path
        self.labelling_strategy = labelling_strategy
        super().__init__(source,
                         batch_size=batch_size,
                         nb_frames=nb_frames,
                         transformations=transformations,
                         printer=printer,
                         frame_size=frame_size,
                         dtype=dtype)

    def assert_source_validity(self, source: pd.DataFrame):
        if not isinstance(source, pd.DataFrame):
            raise TypeError('`source` should be a DataFrame')

        if not self.data_path.exists():
            raise FileNotFoundError('`data_path` not found')

    def get_sample_list(self, source: pd.DataFrame, printer=print):
        self.class_label_map = GeneratorUtils.generate_class_to_label_mapper(
            source.iloc[:, 1].unique(), self.labelling_strategy)

        samples = []
        class_count = {}
        for [sample, _class] in source.values.tolist():
            sample_path = self.data_path / str(sample)
            if len(os.listdir(sample_path)) >= self.nb_frames:
                samples.append([sample_path, self.class_label_map[_class]])

            if _class in class_count:
                class_count[_class] += 1
            else:
                class_count[_class] = 1

        printer(f'Sample size: {len(samples)}')
        for k, v in class_count.items():
            printer(f'Class {k}: {v}')

        return samples

    def process_sample(self, sample: Path):
        return pandasgeneratorutils.process_sample(
            sample, self.nb_frames, self.frame_size, self.dtype, self.transformations)
