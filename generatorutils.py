import math
from pathlib import Path
from typing import Tuple

import albumentations as A
import numpy as np
from PIL import Image
from tensorflow.keras.utils import to_categorical


class GeneratorUtils:
    @staticmethod
    def get_sample_images(sample_path: Path) -> np.array:
        return np.array(sorted(img for img in sample_path.iterdir()))

    @staticmethod
    def pick_at_intervals(A: list, max: int, round_op=None):
        if round_op is None:
            round_op = math.floor

        if len(A) == max:
            return A

        step = len(A) / max
        picked = []
        for i in range(1, max + 1):
            picked.append(A[round_op(i * step) - 1])

        return picked

    @staticmethod
    def process_img(img_path: Path, size: Tuple[int, int] = (224, 224)):
        resized_img = Image.open(img_path).resize(size)
        return np.array(resized_img).astype(np.float32)

    @staticmethod
    def augment(img_arrays, transformations: list):
        additional_targets = {f'image{i}': 'image' for i in range(len(img_arrays)-1)}
        transform = A.Compose(
            transformations, additional_targets=additional_targets)
        transformed = transform(
            image=img_arrays[0],
            **{f'image{i}': img for i, img in enumerate(img_arrays[1:])})

        imgs = [transformed['image']]
        for k in additional_targets:
            imgs.append(transformed[k])

        return np.stack(imgs)

    @staticmethod
    def generate_class_to_label_mapper(classes: list,
                                       labelling_strategy: str):
        if labelling_strategy is 'categorical':
            labels = to_categorical(np.arange(len(classes)))
            return dict(zip(sorted(classes), labels))

        if labelling_strategy is 'binary':
            if len(classes) > 2:
                raise Exception('Binary labelling strategy '
                                'cannot be used when the number '
                                'of classes is greater than two')
            return [0, 1]

        raise ValueError(f'{labelling_strategy} is not a '
                         f'valid labelling strategy.')
