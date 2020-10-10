import os
from generator import Generator
from pathlib import Path


class PathGenerator(Generator):
    def __init__(self, source, batch_size=5):

        super().__init__(source, batch_size)

    def assert_source_validity(self, source: Path):
        if not isinstance(source, Path):
            raise TypeError('source must be a pathlib.Path')

        if not source.exists():
            raise FileNotFoundError('source path does not exist')

        if not source.is_dir():
            raise NotADirectoryError('`source` path is not a directory')

    def get_sample_list(self, source) -> list:
        return list(range(15))


    def process_sample(self, sample):
        pass
