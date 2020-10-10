import math
from pathlib import Path
from PIL import Image
from numpy import asarray
from typing import List
import albumentations as A


class GeneratorUtils:
    @staticmethod
    def get_sample_images(sample_path: Path) -> List[Path]:
        return sorted(img for img in sample_path.iterdir())

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
    def img_to_array(img_path: Path):
        return asarray(Image.open(img_path))

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

        return imgs