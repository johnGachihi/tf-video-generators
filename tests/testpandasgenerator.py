import unittest
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
        cls.source = pd.DataFrame([[1, 'a'], [2, 'b'], [3, 'c']])
        generate_fake_dataset(cls.data_path, cls.source)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree('fake_dataset')

    def test_when_source_not_dataframe(self):
        with self.assertRaises(TypeError):
            PandasGenerator('not a DataFrame', Path('images'))

    def test_when_data_path_non_existent(self):
        with self.assertRaises(FileNotFoundError):
            PandasGenerator(pd.DataFrame([]), Path('non existent'))

    """
    More tests for PandasGenerator

    Improve the test below.
    """
    def test__pandas_generator__yields_correctly(self):
        transformations = [A.HorizontalFlip(p=1)]
        nb_frames = 5
        batch_size = 1
        gen = PandasGenerator(self.source,
                              self.data_path,
                              batch_size=batch_size,
                              nb_frames=nb_frames,
                              transformations=transformations)

        expected = []
        for i in range(1, batch_size + 1):
            imgs = GeneratorUtils.get_sample_images(Path(f'fake_dataset/{i}'))
            imgs = GeneratorUtils.pick_at_intervals(imgs, nb_frames)
            imgs = [GeneratorUtils.process_img(img_path)
                    for img_path in imgs]
            imgs = GeneratorUtils.augment(imgs, transformations)
            expected.append(imgs)
        expected = np.stack(expected)

        for sample, _ in gen:
            npt.assert_equal(actual=sample, desired=expected)
            break



def generate_fake_dataset(path: Path, labels: pd.DataFrame):
    if not path.exists():
        path.mkdir()
    for sample_name, _ in labels.values.tolist():
        sample_dir = path / str(sample_name)
        if not sample_dir.exists():
            sample_dir.mkdir()
        for img in Path('images').iterdir():
            shutil.copy(img, sample_dir)