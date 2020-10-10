import unittest
from pandasgenerator import PandasGenerator
import pandas as pd
from pathlib import Path
import shutil


class TestPandasGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        generate_fake_dataset(Path('fake_dataset'),
                              pd.DataFrame([[1, 'a'], [2, 'b'], [3, 'c']]))

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree('fake_dataset')

    def test_when_source_not_dataframe(self):
        with self.assertRaises(TypeError):
            PandasGenerator('not a DataFrame', Path('images'))

    def test_when_data_path_non_existent(self):
        with self.assertRaises(FileNotFoundError):
            PandasGenerator(pd.DataFrame([]), Path('non existent'))

    def test_samples_generated_correctly(self):
        data = pd.DataFrame([[1, 'a'],
                             [2, 'b'],
                             [3, 'c']])
        gen = PandasGenerator(data, Path('images'))

        expected = [[Path('images/1'), 'a'],
                    [Path('images/2'), 'b'],
                    [Path('images/3'), 'c']]

        self.assertEqual(expected, gen.samples)

    def test_get_images(self):
        sample = Path('fake_dataset/1')
        expected_sample_imgs = [sample for sample in sample.iterdir()]
        actual_sample_imgs = PandasGenerator.get_images(sample, 3)
        self.assertCountEqual(expected_sample_imgs, actual_sample_imgs)

    def test_pick_at_intervals(self):
        picked = PandasGenerator.pick_at_intervals(list(range(5)), 5)
        self.assertEqual(list(range(5)), picked)

        picked = PandasGenerator.pick_at_intervals([1, 2, 3, 4], 3)
        self.assertEqual([1, 2, 4], picked)

        picked = PandasGenerator.pick_at_intervals([1, 2, 3, 4, 5, 6], 3)





def generate_fake_dataset(path: Path, labels: pd.DataFrame):
    if not path.exists():
        path.mkdir()
    for sample_name, _ in labels.values.tolist():
        sample_dir = path / str(sample_name)
        if not sample_dir.exists():
            sample_dir.mkdir()
        for img in Path('images').iterdir():
            shutil.copy(img, sample_dir)