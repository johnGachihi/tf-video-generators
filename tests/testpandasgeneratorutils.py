import os
import shutil
import unittest
from pathlib import Path
from random import randrange
import numpy.testing as npt
import albumentations as A

from generatorutils import GeneratorUtils
from pandasgeneratorutils import process_sample


class TestPandasGeneratorUtils(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.fake_sample = Path('fake-sample')
        get_fake_sample(cls.fake_sample, 4)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree('fake-sample')

    def test__process_sample__without_transformations(self):
        nb_frames = 3
        frame_size = (20, 20)

        img_paths = GeneratorUtils.pick_at_intervals(
            GeneratorUtils.get_sample_images(self.fake_sample),
            nb_frames)
        expected = [GeneratorUtils.process_img(img_path, frame_size)
                      for img_path in img_paths]

        actual = process_sample(self.fake_sample, nb_frames, frame_size)

        npt.assert_equal(desired=expected, actual=actual)

    def test__process_sample__with_transformations(self):
        nb_frames = 3
        frame_size = (20, 20)
        transformations = [A.HorizontalFlip(p=1)]

        img_paths = GeneratorUtils.pick_at_intervals(
            GeneratorUtils.get_sample_images(self.fake_sample),
            nb_frames)
        img_arrays = [GeneratorUtils.process_img(img_path, frame_size)
                    for img_path in img_paths]
        expected = GeneratorUtils.augment(img_arrays, transformations)

        actual = process_sample(self.fake_sample, nb_frames, frame_size, transformations=transformations)

        npt.assert_equal(desired=expected, actual=actual)


"""
Creates fake sample folder
Structure:
- <fake_sample_name>
    - 1.png
    - 2.png
    - 3.png
    - 4.png
    - 5.png
    - ...
"""
def get_fake_sample(path: Path, nb_frames: int):
    if not path.exists():
        path.mkdir()

    imgs_path = Path('images')
    imgs = os.listdir(imgs_path)
    nb_images = len(imgs)

    for frame in range(nb_frames):
        img_path = imgs_path / imgs[randrange(0, nb_images)]
        shutil.copy(img_path, path / f'{frame}.png')

