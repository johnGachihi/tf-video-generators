import math
import unittest
from unittest.mock import Mock
from pandasgenerator import PandasGenerator
import pandas as pd
from pathlib import Path
import shutil
import albumentations as A
from generatorutils import GeneratorUtils
import numpy as np
import numpy.testing as npt


class TestPandasGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data_path = Path('fake_dataset')
        cls.source = pd.DataFrame([[1, 'a'], [2, 'b'], [3, 'c'], [4, 'c']])
        cls.nb_samples = len(cls.source.index)
        generate_fake_dataset(cls.data_path, cls.source)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree('fake_dataset')

    def test_when_source_not_dataframe(self):
        with self.assertRaises(TypeError):
            PandasGenerator('not a DataFrame', Path('images'))

    def test_when_data_path_non_existent(self):
        with self.assertRaises(FileNotFoundError):
            PandasGenerator(self.source, Path('non existent'))

    def test__pandas_generator__yields_sample_images_correctly(self):
        transformations = [A.HorizontalFlip(p=1)]
        nb_frames = 5
        batch_size = self.nb_samples
        frame_size = (10, 10)
        gen = PandasGenerator(self.source,
                              self.data_path,
                              batch_size=batch_size,
                              nb_frames=nb_frames,
                              transformations=transformations,
                              frame_size=frame_size)

        expected = []
        for i in range(1, batch_size + 1):
            imgs = GeneratorUtils.get_sample_images(Path(f'fake_dataset/{i}'))
            imgs = GeneratorUtils.pick_at_intervals(imgs, nb_frames, math.floor)
            imgs = [GeneratorUtils.process_img(img_path, frame_size)
                    for img_path in imgs]
            imgs = GeneratorUtils.augment(imgs, transformations)
            expected.append(imgs)
        expected = np.stack(expected)

        sample, _ = gen.__getitem__(0)
        npt.assert_equal(actual=sample, desired=expected)

    def test__pandas_generator__batch_size_yielded_as_specified(self):
        batch_size = self.nb_samples
        gen = PandasGenerator(self.source,
                              self.data_path,
                              nb_frames=2,
                              batch_size=batch_size)

        samples, labels = gen.__getitem__(0)
        self.assertEqual(batch_size, samples.shape[0])
        self.assertEqual(batch_size, labels.shape[0])

    def test__pandas_generator__nb_frames_yielded_as_specified(self):
        nb_frames = 2
        gen = PandasGenerator(self.source,
                              self.data_path,
                              nb_frames=nb_frames,
                              batch_size=1)

        samples, _ = gen.__getitem__(0)
        self.assertEqual(nb_frames, samples.shape[1])

    def test__pandas_generator__number_of_batches_yielded(self):
        batch_size = 2
        gen = PandasGenerator(self.source,
                              self.data_path,
                              nb_frames=5,
                              batch_size=batch_size)

        batches = []
        for samples, _ in gen:
            batches.append(samples)

        self.assertEqual(math.ceil(self.nb_samples / batch_size), len(batches))

    def test__pandas_generator__labels(self):
        classes = self.source.loc[:, 1].unique()
        class_label_map = GeneratorUtils.generate_class_to_label_mapper(
            classes, 'categorical')
        expected = list(map(lambda c: class_label_map[c], self.source.iloc[:, 1].values))

        gen = PandasGenerator(self.source,
                              self.data_path,
                              nb_frames=5,
                              batch_size=self.nb_samples)

        _, labels = gen.__getitem__(0)
        npt.assert_equal(actual=labels, desired=expected)

    def test__pandas_generator__ignores_samples_that_have_less_frames_than_nb_frames(self):
        gen = PandasGenerator(self.source,
                              self.data_path,
                              nb_frames=6,
                              batch_size=1)

        batches = []
        for samples, _ in gen:
            batches.append(samples)

        self.assertEqual(0, len(batches))

    def test__pandas_generator__prints_suitable_samples(self):
        mock_printer = Mock()
        PandasGenerator(self.source,
                        self.data_path,
                        nb_frames=5,
                        batch_size=1,
                        printer=mock_printer)

        mock_printer.assert_any_call(f'Sample size: {self.nb_samples}')
        for _class, count in self.source.iloc[:, 1].value_counts().items():
            mock_printer.assert_any_call(f'Class {_class}: {count}')

"""
Creates fake dataset folder
Structure:
- fake_dataset
    - 1
        - 1.png
        - 2.png
        - 3.png
        - 4.png
        - 5.png
    - 2
        - 1.png
        - 2.png
        - 3.png
        - 4.png
        - 5.png
    - 3
        - ...
"""
def generate_fake_dataset(path: Path, labels: pd.DataFrame):
    if not path.exists():
        path.mkdir()
    for sample_name, _ in labels.values.tolist():
        sample_dir = path / str(sample_name)
        if not sample_dir.exists():
            sample_dir.mkdir()
        for img in Path('images').iterdir():
            shutil.copy(img, sample_dir)