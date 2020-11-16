import shutil
import unittest
from pathlib import Path
import pandas as pd
import numpy as np
import numpy.testing as npt

from pandaspredictiongenerator import PandasPredictionGenerator
import pandasgeneratorutils


class TestPandasPredictionGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data_path = Path('fake_dataset')
        cls.source = pd.DataFrame([1, 2, 3, 4])
        cls.nb_samples = len(cls.source.index)
        generate_fake_dataset(cls.data_path, cls.source.values)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.data_path)

    def test__pandas_prediction_generator__yields_correct_output(self):
        batch_size = self.nb_samples
        nb_frames = 3
        frame_size = (50, 50)
        gen = PandasPredictionGenerator(
            self.source,
            self.data_path,
            batch_size=batch_size,
            nb_frames=3,
            frame_size=frame_size)

        expected = []
        for i in range(1, batch_size+1):
            sample = pandasgeneratorutils.process_sample(
                self.data_path / f'{i}',
                nb_frames=nb_frames,
                frame_size=frame_size)
            expected.append(sample)
        expected = np.stack(expected)

        actual = gen.__getitem__(0)

        npt.assert_equal(actual, expected)







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

def generate_fake_dataset(path: Path, samples):
    if not path.exists():
        path.mkdir()
    for sample_name in samples:
        sample_dir = path / str(sample_name.item())

        if not sample_dir.exists():
            sample_dir.mkdir()
        for img in Path('images').iterdir():
            shutil.copy(img, sample_dir)
