from generator import Generator
import pandas as pd
from pathlib import Path
from generatorutils import GeneratorUtils


class PandasGenerator(Generator):
    def __init__(
            self,
            source: pd.DataFrame,
            data_path: Path,
            batch_size=5,
            nb_frames=8,
            transformations=None,
            labelling_strategy='categorical'
    ):
        self.data_path = data_path
        self.labelling_strategy = labelling_strategy
        super().__init__(source,
                         batch_size=batch_size,
                         nb_frames=nb_frames,
                         transformations=transformations)

    def assert_source_validity(self, source: pd.DataFrame):
        if not isinstance(source, pd.DataFrame):
            raise TypeError('`source` should be a DataFrame')

        if not self.data_path.exists():
            raise FileNotFoundError('`data_path` not found')

    def get_sample_list(self, source: pd.DataFrame):  # Is the source parameter necessary
        class_label_map = GeneratorUtils.generate_class_to_label_mapper(
            source.loc[:, 1].values.tolist(), self.labelling_strategy)

        return [[self.data_path / str(sample), class_label_map[_class]]
                for [sample, _class] in source.values.tolist()]

    def process_sample(self, sample: Path):
        img_paths = GeneratorUtils.pick_at_intervals(
            GeneratorUtils.get_sample_images(sample),
            self.nb_frames)  # Add third param

        img_arrays = [GeneratorUtils.process_img(img_path)
                      for img_path in img_paths]

        if self.transformations is None:
            return img_arrays
        else:
            return GeneratorUtils.augment(img_arrays, self.transformations)
