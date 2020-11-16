from abc import abstractmethod
from tensorflow.keras.utils import Sequence


class Generator(Sequence):
    def __init__(self,
                 source,
                 batch_size=5,
                 nb_frames=10,
                 printer=print,
                 frame_size=(224, 224)):
        self.assert_source_validity(source)

        self.nb_frames = nb_frames
        self.batch_size = batch_size
        self.frame_size = frame_size
        self.samples = self.get_sample_list(source, printer)

    @abstractmethod
    def assert_source_validity(self, source):
        pass

    @abstractmethod
    def get_sample_list(self, source, printer=print) -> list:
        """
        Retrieve a list of samples with their labels

        :param source: The source or an identifier of the
         source data eg. a dataframe, folder path, etc
        :param printer: Function to be used to print sample
         stats.
        """
        pass

    @abstractmethod
    def process_sample(self, sample):
        """
        Process a sample. To return sample data
        :param sample: Process
        :return:
        """
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
