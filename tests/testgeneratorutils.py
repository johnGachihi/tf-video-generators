from pathlib import Path
from unittest import TestCase
from generatorutils import GeneratorUtils
import math
import pandas as pd
import shutil
from PIL import Image
import numpy as np
from numpy import asarray
import albumentations as A


class TestGeneratorUtils(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        generate_fake_dataset(Path('fake_dataset'),
                              pd.DataFrame([[1, 'a'], [2, 'b'], [3, 'c']]))

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree('fake_dataset')


    def test__pick_at_intervals__when_max_equals_list_length(self):
        expected = [1, 2, 3, 4]
        actual = GeneratorUtils.pick_at_intervals(expected, 4)
        self.assertEqual(expected, actual)

    def test__pick_at_intervals__when_required_elements_less_than_available_using_floor(self):
        actual = GeneratorUtils.pick_at_intervals([1, 2, 3, 4], 3, math.floor)
        self.assertEqual([1, 2, 4], actual)

    def test__pick_at_intervals__when_required_elements_less_than_available_using_ceil(self):
        actual = GeneratorUtils.pick_at_intervals([1, 2, 3, 4], 3, math.ceil)
        self.assertEqual([2, 3, 4], actual)

    def test__get_sample_images(self):
        sample_path = Path('fake_dataset/1')
        expected = [Path('fake_dataset/1/1.png'),
                    Path('fake_dataset/1/2.png'),
                    Path('fake_dataset/1/3.png'),
                    Path('fake_dataset/1/4.png'),
                    Path('fake_dataset/1/5.png')]
        actual = GeneratorUtils.get_sample_images(sample_path)

        self.assertEqual(expected, actual)

    def test__img_to_array(self):
        img_path = Path('fake_dataset/1/1.png')
        expected = asarray(Image.open(img_path))
        actual = GeneratorUtils.img_to_array(img_path)
        self.assertTrue(np.array_equal(expected, actual))

    def test__augment_imgs__transforms_imgs(self):
        sample_path = Path('fake_dataset/1')
        transformations = [A.HorizontalFlip(p=1)]
        transform = A.Compose(
            transformations,
            additional_targets={f'image{i}': 'image' for i in range(4)})
        imgs = GeneratorUtils.get_sample_images(sample_path)
        imgs = [GeneratorUtils.img_to_array(img) for img in imgs]
        expected = transform(image=imgs[0], **{f'image{i}': img for i, img in enumerate(imgs[1:])})

        actual = GeneratorUtils.augment(imgs, transformations)

        self.assertTrue(np.array_equal(expected['image'], actual[0]))
        self.assertTrue(np.array_equal(expected['image0'], actual[1]))
        self.assertTrue(np.array_equal(expected['image1'], actual[2]))
        self.assertTrue(np.array_equal(expected['image2'], actual[3]))
        self.assertTrue(np.array_equal(expected['image3'], actual[4]))

    def test__augment_imgs__maintains_order(self):
        sample_path = Path('fake_dataset/1')
        transformations = [A.HorizontalFlip(p=1)]
        transform = A.Compose(transformations)
        imgs = GeneratorUtils.get_sample_images(sample_path)
        imgs = [GeneratorUtils.img_to_array(img) for img in imgs]

        actual = GeneratorUtils.augment(imgs, transformations)

        transformed = transform(image=imgs[0])
        self.assertTrue(np.array_equal(transformed['image'], actual[0]))

        transformed = transform(image=imgs[1])
        self.assertTrue(np.array_equal(transformed['image'], actual[1]))

        transformed = transform(image=imgs[2])
        self.assertTrue(np.array_equal(transformed['image'], actual[2]))

        transformed = transform(image=imgs[3])
        self.assertTrue(np.array_equal(transformed['image'], actual[3]))

        transformed = transform(image=imgs[4])
        self.assertTrue(np.array_equal(transformed['image'], actual[4]))

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