from pathlib import Path
from unittest import TestCase
from generatorutils import GeneratorUtils
import math
import pandas as pd
import shutil
from PIL import Image
import numpy as np
from numpy import asarray
import numpy.testing as npt
import albumentations as A
from tensorflow.keras.utils import to_categorical



class TestGeneratorUtils(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data_path = Path('fake_dataset')
        cls.source = pd.DataFrame([[1, 'a'], [2, 'b'], [3, 'c']])
        generate_fake_dataset(cls.data_path, cls.source)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.data_path)

    def test__pick_at_intervals__when_max_equals_list_length(self):
        expected = [1, 2, 3, 4]
        actual = GeneratorUtils.pick_at_intervals(expected, 4, math.floor)
        self.assertEqual(expected, actual)

    def test__pick_at_intervals__when_required_elements_less_than_available_using_floor(self):
        actual = GeneratorUtils.pick_at_intervals([1, 2, 3, 4], 3, math.floor)
        self.assertEqual([1, 2, 4], actual)

    def test__pick_at_intervals__when_required_elements_less_than_available_using_ceil(self):
        actual = GeneratorUtils.pick_at_intervals([1, 2, 3, 4], 3, math.ceil)
        self.assertEqual([2, 3, 4], actual)

    def test__get_sample_images(self):
        sample_path = Path('fake_dataset/1')
        expected = np.array([
            Path('fake_dataset/1/1.png'),
            Path('fake_dataset/1/2.png'),
            Path('fake_dataset/1/3.png'),
            Path('fake_dataset/1/4.png'),
            Path('fake_dataset/1/5.png')])
        actual = GeneratorUtils.get_sample_images(sample_path)

        npt.assert_equal(actual, expected)

    def test__process_img__resizes_img(self):
        img_path = Path('fake_dataset/1/1.png')
        img = GeneratorUtils.process_img(img_path, (10, 10))

        self.assertEqual((10, 10), img.shape[:2])

    def test__process_img__converts_img_to_numpy_array(self):
        img_path = Path('fake_dataset/1/1.png')
        img = GeneratorUtils.process_img(img_path, (10, 10))

        self.assertIsInstance(img, np.ndarray)

    def test__augment_imgs__returns_numpy_array(self):
        imgs = GeneratorUtils.get_sample_images(Path('fake_dataset/1'))
        imgs = [GeneratorUtils.process_img(img) for img in imgs]
        augmented_imgs = GeneratorUtils.augment(imgs, [A.HorizontalFlip()])

        self.assertIsInstance(augmented_imgs, np.ndarray)

    def test__augment_imgs__transforms_imgs(self):
        sample_path = Path('fake_dataset/1')
        transformations = [A.HorizontalFlip(p=1)]
        transform = A.Compose(
            transformations,
            additional_targets={f'image{i}': 'image' for i in range(4)})
        imgs = GeneratorUtils.get_sample_images(sample_path)
        imgs = [GeneratorUtils.process_img(img) for img in imgs]
        expected = transform(image=imgs[0], **{f'image{i}': img for i, img in enumerate(imgs[1:])})

        actual = GeneratorUtils.augment(imgs, transformations)

        npt.assert_equal(expected['image'], actual[0])
        npt.assert_equal(expected['image0'], actual[1])
        npt.assert_equal(expected['image1'], actual[2])
        npt.assert_equal(expected['image2'], actual[3])
        npt.assert_equal(expected['image3'], actual[4])

    def test__augment_imgs__maintains_order(self):
        sample_path = Path('fake_dataset/1')
        transformations = [A.HorizontalFlip(p=1)]
        transform = A.Compose(transformations)
        imgs = GeneratorUtils.get_sample_images(sample_path)
        imgs = [GeneratorUtils.process_img(img) for img in imgs]

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

    def test__generate_class_to_label_mapper__when_strategy_is_categorical(self):
        classes = ['a', 'b', 'c']
        labels = to_categorical(np.arange(len(classes)))
        class_to_label_map = dict(zip(classes, labels))

        actual = GeneratorUtils.generate_class_to_label_mapper(
            classes, 'categorical')

        npt.assert_equal(class_to_label_map, actual)

    def test__generate_class_to_label_mapper__when_strategy_is_binary_with_2_classes(self):
        classes = ['a', 'b']

        actual = GeneratorUtils.generate_class_to_label_mapper(
            classes, 'binary')

        npt.assert_equal(desired=[0, 1], actual=actual)

    def test__generate_class_to_label_mapper__when_strategy_is_binary_with_more_than_two_classes(self):
        with self.assertRaises(Exception):
            GeneratorUtils.generate_class_to_label_mapper(
                [0, 1, 2], 'binary')

    def test__generate_class_to_label_mapper__when_strategy_is_neither_binary_nor_categorical(self):
        with self.assertRaises(ValueError):
            GeneratorUtils.generate_class_to_label_mapper(
                [0, 1], 'not a valid strategy')

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