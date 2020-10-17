from typing import List

from generator import Generator
import pandas as pd
from pathlib import Path
from generatorutils import GeneratorUtils
import os


class PandasGenerator(Generator):
    def __init__(
            self,
            source: pd.DataFrame,
            data_path: Path,
            batch_size=5,
            nb_frames=8,
            transformations=None,
            labelling_strategy='categorical',
            printer=print,
            frame_size=(224, 224)
    ):
        self.data_path = data_path
        self.labelling_strategy = labelling_strategy
        super().__init__(source,
                         batch_size=batch_size,
                         nb_frames=nb_frames,
                         transformations=transformations,
                         printer=printer,
                         frame_size=frame_size)

    def assert_source_validity(self, source: pd.DataFrame):
        if not isinstance(source, pd.DataFrame):
            raise TypeError('`source` should be a DataFrame')

        if not self.data_path.exists():
            raise FileNotFoundError('`data_path` not found')

    def get_sample_list(self, source: pd.DataFrame, printer=print):
        class_label_map = GeneratorUtils.generate_class_to_label_mapper(
            source.iloc[:, 1].unique(), self.labelling_strategy)

        samples = []
        class_count = {}
        for [sample, _class] in source.values.tolist():
            sample_path = self.data_path / str(sample)
            if len(os.listdir(sample_path)) >= self.nb_frames:
                samples.append([sample_path, class_label_map[_class]])

            if _class in class_count:
                class_count[_class] += 1
            else:
                class_count[_class] = 1

        printer(f'Sample size: {len(samples)}')
        for k, v in class_count.items():
            printer(f'Class {k}: {v}')

        return samples

    def process_sample(self, sample: Path):
        img_paths = GeneratorUtils.pick_at_intervals(
            GeneratorUtils.get_sample_images(sample),
            self.nb_frames)  # Add third param - random round op

        img_arrays = [GeneratorUtils.process_img(img_path, self.frame_size)
                      for img_path in img_paths]

        if self.transformations is None:
            return img_arrays
        else:
            return GeneratorUtils.augment(img_arrays, self.transformations)
