"""
Dataset and dataloader for imsitu experiments.

This allows us to:

1) Finetune on Imsitu
2) Finetune on a zero shot setting
"""
import spacy
import torch
import os
from config import IMSITU_TRAIN_LIST, IMSITU_VAL_LIST, IMSITU_TEST_LIST, IMSITU_IMGS
from torchvision.transforms import Scale, RandomCrop, CenterCrop, ToTensor, Normalize, Compose
from PIL import Image
from data.attribute_loader import Attributes
from collections import namedtuple
from torch.autograd import Variable

LISTS = {
    'train': IMSITU_TRAIN_LIST,
    'val': IMSITU_VAL_LIST,
    'test': IMSITU_TEST_LIST,
}


def _load_imsitu_file(mode):
    """
    Helper fn that loads imsitu file
    :param fn:
    :return:
    """
    if mode not in LISTS:
        raise ValueError("Invalid mode {}, must be train val or test".format(mode))

    imsitu_ind_to_label = {}
    dps = []
    with open(LISTS[mode], 'r') as f:
        for row in f.read().splitlines():
            fn_ext = row.split(' ')[0]
            label = fn_ext.split('_')[0]  # This has "ing" on it, so we can't use it for the word
                                          # label. But needed to construct the filename
            ind = int(row.split(' ')[1])

            fn = os.path.join(IMSITU_IMGS, label, fn_ext)
            imsitu_ind_to_label[ind] = label
            dps.append((fn, ind))
    return dps


class ImSitu(torch.utils.data.Dataset):
    def __init__(self,
                 use_train_verbs=False,
                 use_val_verbs=False,
                 use_test_verbs=False,
                 use_train_images=False,
                 use_val_images=False,
                 use_test_images=False,
                 ):
        self.use_train_verbs = use_train_verbs
        self.use_val_verbs = use_val_verbs
        self.use_test_verbs = use_test_verbs

        if not (self.use_train_verbs or self.use_val_verbs or self.use_test_verbs):
            raise ValueError("No verbs selected!")

        self.use_train_images = use_train_images
        self.use_val_images = use_val_images
        self.use_test_images = use_test_images

        if not (self.use_train_verbs or self.use_val_verbs or self.use_test_verbs):
            raise ValueError("No images selected!")

        self.attributes = Attributes(use_train=self.use_train_verbs, use_val=self.use_val_verbs,
                                     use_test=self.use_test_verbs, imsitu_only=True)

        self.examples = []
        for mode, to_use in zip(
            ['train', 'val', 'test'], 
            [self.use_train_images, self.use_val_images, self.use_test_images],
        ):
            self.examples += [(fn, self.attributes.ind_perm[ind])
                              for fn, ind in _load_imsitu_file(mode)
                              if ind in self.attributes.ind_perm]

        self.transform = transform(is_train=not self.use_test_verbs)

    def __getitem__(self, index):
        fn, ind = self.examples[index]
        img = self.transform(Image.open(fn).convert('RGB'))
        return img, ind

    @classmethod
    def splits(cls, zeroshot_transfer=False, **kwargs):
        """
        Gets splits
        :param zeroshot_transfer: True if we're transferring to zeroshot classes
        :return: train, val, test datasets
        """

        if zeroshot_transfer:
            train_cls = cls(use_train_verbs=True, use_train_images=True, use_val_images=True)
            val_cls = cls(use_val_verbs=True, use_train_images=True, use_val_images=True)
            test_cls = cls(use_test_verbs=True, use_test_images=True)
        else:
            train_cls = cls(use_train_verbs=True, use_train_images=True)
            val_cls = cls(use_train_verbs=True, use_val_images=True)
            test_cls = cls(use_train_verbs=True, use_test_images=True)
        return train_cls, val_cls, test_cls

    def __len__(self):
        return len(self.examples)

Batch = namedtuple('Batch', ['img', 'label'])


class CudaDataLoader(torch.utils.data.DataLoader):
    """
    Iterates through the data, but also loads everything as a (cuda) variable
    """

    def _load(self, item):
        img = Variable(item[0])
        label = Variable(item[1])

        if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()

        return Batch(img, label)

    def __iter__(self):
        return (self._load(x) for x in super(CudaDataLoader, self).__iter__())


def transform(is_train=True, normalize=True):
    """
    Returns a transform object
    """
    filters = []
    filters.append(Scale(256))

    if is_train:
        filters.append(RandomCrop(224))
    else:
        filters.append(CenterCrop(224))

    filters.append(ToTensor())

    if normalize:
        filters.append(Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]))
    return Compose(filters)


def collate_fn(data):
    imgs, labels = zip(*data)
    imgs = torch.stack(imgs, 0)
    labels = torch.LongTensor(labels)
    return imgs, labels


if __name__ == '__main__':
    train, val, test = ImSitu.splits()
    train_dl = CudaDataLoader(
        dataset=train,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
