import os

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden

from .imagenet import ImageNet
from .oxford_pets import OxfordPets

TO_BE_IGNORED = ["README.txt"]


@DATASET_REGISTRY.register()
class ImageNetR(DatasetBase):
    """ImageNet-R(endition).

    This dataset is used for testing only.
    """

    dataset_dir = "imagenet-rendition"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "imagenet-r")

        text_file = os.path.join(self.dataset_dir, "classnames.txt")
        self.all_classnames = ImageNet.read_classnames(text_file)

        data = self.read_data(self.all_classnames)

        
        # ordered_classnames = []
        # idx = 0
        # while idx < 200:
        #     for arg in (data,):
        #         for item in arg:
        #             if item.label == idx:
        #                 ordered_classnames.append(item.classname)
        #                 idx += 1
        # print(ordered_classnames[100:200])
        # print(len(ordered_classnames))
        # assert(0)

        self.all_classattrs = cfg.DATASET.ATTR
        self.all_classbias = cfg.DATASET.BIAS
        subsample = 'all'
        self.classattrs = OxfordPets.get_attrs(self.all_classattrs, subsample)
        self.classbias = OxfordPets.get_attrs(self.all_classbias, subsample)

        super().__init__(train_x=data, test=data)

    def read_data(self, classnames):
        image_dir = self.image_dir
        folders = listdir_nohidden(image_dir, sort=True)
        folders = [f for f in folders if f not in TO_BE_IGNORED]
        items = []

        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(image_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(image_dir, folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items