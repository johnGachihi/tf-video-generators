import math
import os
import numpy as np
import random
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import Sequence, to_categorical
import albumentations as A


class CustomGenerator(Sequence):
    def __init__(
            self,
            data_path,
            nbframes,
            target_size=(150, 150),
            albumentation=None,
            batch_size=32,
            labeling_strategy=None,
            val_split=0.2,
            test_split=0.2,
            data=None
    ):
        self.__assert_datapath_isvalid(data_path)

        self.data_path = data_path
        self.batch_size = batch_size
        self.nbframes = nbframes
        self.target_size = target_size
        self.labeling_strategy = labeling_strategy
        self.val_split = val_split
        self.test_split = test_split
        self.albumentation = albumentation
        self.augmentor = self.__get_augmentor()
        self.classes = self.__getclasses()
        self.class_to_label_map = self.__get_class_to_label_map()

        if data is None:
            self.data = self.__getdata()
            self.__splitdata()
        else:
            self.data = data

    def __assert_datapath_isvalid(self, datapath):
        if not os.path.exists(datapath):
            raise Exception(f'{datapath} does not exist')

    def __get_augmentor(self):
        resize = A.Resize(self.target_size[1], self.target_size[0])
        transforms = [resize]
        if self.albumentation:
            transforms = transforms * self.albumentation.transforms
        targets = {f'image{i}': 'image' for i in range(1, self.nbframes + 1)}
        return A.Compose(transforms, additional_targets=targets)

    def __getclasses(self):
        return [dir for dir in os.listdir(self.data_path)
                if os.path.isdir(os.path.join(self.data_path, dir))]

    def __get_class_to_label_map(self):
        return dict(zip(self.__getclasses(), self.__getlabels()))

    def __getlabels(self):
        if self.labeling_strategy is None:
            self.__guess_labeling_strategy()

        if self.labeling_strategy is 'categorical':
            labels = to_categorical(list(range(len(self.classes))))
        elif self.labeling_strategy is 'binary':
            labels = [0, 1]

        return labels

    def __guess_labeling_strategy(self):
        if len(self.classes) is 2:
            self.labeling_strategy = 'binary'
        else:
            self.labeling_strategy = 'categorical'

    def __getdata(self):
        data = [[sample, self.class_to_label_map[cls]]
                for cls in self.__getclasses()
                for sample in self.__getclass_samples(cls)]
        random.shuffle(data)
        return data

    def __getclass_samples(self, cls):
        samples = []
        for sample in os.listdir(os.path.join(self.data_path, cls)):
            samplepath = os.path.join(self.data_path, cls, sample)
            samplepath_isdir = os.path.isdir(samplepath)
            sampleframes = len(os.listdir(samplepath))  # Logic issue
            if samplepath_isdir and sampleframes >= self.nbframes:
                samples.append(os.path.join(cls, sample))
        print(f'Found {len(samples)} samples for {cls}')
        return samples

    def __splitdata(self):
        valsize = math.floor(self.val_split * len(self.data))
        testsize = math.floor(self.test_split * len(self.data))
        self.val_data = self.data[:valsize]
        self.test_data = self.data[-testsize:]
        self.data = self.data[valsize:-testsize]

    def __len__(self):
        return math.ceil(len(self.data) / self.batch_size)

    def __getitem__(self, idx):
        batch_data = self.data[idx * self.batch_size: (idx + 1) * self.batch_size]
        X, y = [], []
        for sample, label in batch_data:
            sample_imgs = self.__getsample_images(sample)
            X.append(sample_imgs)
            y.append(label)

        return np.array(X), np.array(y)

    def __getsample_images(self, sample):
        sample_path = os.path.join(self.data_path, sample)
        img_paths = self.__getimagepaths(sample_path)
        if len(img_paths) > self.nbframes:
            img_paths = self.__pickimages(img_paths)
        imgs = np.array([self.__img_to_array(img_path) for img_path in img_paths])
        return self.__augment_imgs(imgs)

    def __getimagepaths(self, sample_path):
        return [os.path.join(sample_path, img_path)
                for img_path in sorted(os.listdir(sample_path))]

    def __pickimages(self, img_paths):
        if (len(img_paths) + 2) > self.nbframes:
            img_paths = img_paths[1:-1]

        if len(img_paths) == self.nbframes:
            return np.array(img_paths)

        step = len(img_paths) / self.nbframes
        picked_imgs = []
        for i in range(1, self.nbframes + 1):
            picked_imgs.append(img_paths[math.floor(i * step) - 1])
        return np.array(picked_imgs)

    def __img_to_array(self, img_path):
        img = load_img(img_path)
        img_arr = img_to_array(img)
        img_arr = np.array(img_arr)
        return img_arr

    def __augment_imgs(self, imgs):
        other_targets = {f'image{i + 1}': img for i, img in enumerate(imgs[1:])}
        transformed = self.augmentor(image=imgs[0], **other_targets)
        transformed_imgs = [transformed['image']]
        for target in other_targets:
            transformed_imgs.append(transformed[target])
        return np.stack(transformed_imgs)

    def get_valgen(self):
        return CustomGenerator(
            self.data_path,
            self.nbframes,
            self.target_size,
            albumentation=self.albumentation,
            batch_size=self.batch_size,
            labeling_strategy=self.labeling_strategy,
            val_split=None,
            test_split=None,
            data=self.val_data)

    def get_testgen(self):
        return CustomGenerator(
            self.data_path,
            self.nbframes,
            self.target_size,
            albumentation=self.albumentation,
            batch_size=self.batch_size,
            labeling_strategy=self.labeling_strategy,
            val_split=None,
            test_split=None,
            data=self.test_data)