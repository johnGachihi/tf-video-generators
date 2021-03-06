from pathlib import Path
from typing import Tuple

from generatorutils import GeneratorUtils
import numpy as np


def process_sample(sample: Path,
                   nb_frames: int,
                   frame_size: Tuple[int, int],
                   dtype=np.uint8,
                   transformations: list = None):
    img_paths = GeneratorUtils.pick_at_intervals(
        GeneratorUtils.get_sample_images(sample),
        nb_frames)  # Add third param - random round op

    img_arrays = [GeneratorUtils.process_img(img_path, frame_size, dtype)
                  for img_path in img_paths]

    if transformations is None:
        return img_arrays
    else:
        return GeneratorUtils.augment(img_arrays, transformations)
