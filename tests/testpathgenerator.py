import unittest
import os
from pathgenerator import PathGenerator
from pathlib import Path
import shutil
import random
import math
from customgenerator import CustomGenerator


class TestGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.classes = 2
        cls.samples = 3
        generate_fake_dataset(Path('../assets/fake_dataset'), 3, 5, 6)

    @classmethod
    def tearDownClass(cls) -> None:
        # shutil.rmtree(Path('../fake_dataset'))
        pass

    def test_source_not_path(self):
        with self.assertRaises(TypeError):
            PathGenerator('string')

    def test_non_existing_source_path(self):
        with self.assertRaises(FileNotFoundError):
            PathGenerator(Path('/non-existent-source-path'))

    def test_non_dir_source_path(self):
        with open('example.txt', 'w'):
            pass

        with self.assertRaises(NotADirectoryError):
            PathGenerator(Path('example.txt'))

        os.remove('example.txt')

    # def test_get_sample_list_len(self):
    #     gen = PathGenerator(Path('../assets/fake_dataset'), batch_size=3)
    #     self.assertEqual(math.ceil((3*5) / 3), gen.__len__())
    #
    # def test_path_generator(self):
    #     gen = PathGenerator(Path('../assets/fake_dataset'), batch_size=3)
        # for/


def generate_fake_dataset(path: Path,
                          nb_classes: int,
                          sample_size: int,
                          imgs_per_sample: int):
    imgs_dir = Path('../assets/images')
    imgs = [img for img in imgs_dir.iterdir()]

    if not path.exists():
        path.mkdir()

    for c in range(nb_classes):
        class_dir = path / f'class_{c}'
        if not class_dir.exists():
            class_dir.mkdir()
        for sample in range(sample_size):
            sample_dir = class_dir / f'sample_{sample}'
            if not sample_dir.exists():
                sample_dir.mkdir()
            for img in range(imgs_per_sample):
                img = imgs[random.randrange(0, len(imgs))]
                shutil.copy(img, sample_dir)
