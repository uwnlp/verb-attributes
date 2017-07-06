"""
Configuration file!
"""
import os
from argparse import ArgumentParser

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(ROOT_PATH, 'data')
IMSITU_PATH = '/home/rowan/datasets/imsitu'

def path(fn):
    return os.path.join(DATA_PATH, fn)


def vwa_path(fn):
    return os.path.join(DATA_PATH, 'VerbsWithAttributes', fn)


def imsitu_path(fn):
    return os.path.join(DATA_PATH, 'VerbsWithAttributes', fn)



ATTRIBUTES_PATH = vwa_path('attributes.csv')
ATTRIBUTES_SPLIT = vwa_path('attributes_split.csv')
DICTIONARY_PATH = path('dictionary_challenge.pkl')

IMSITU_LABELS = imsitu_path('OpenFrame500.tab')
IMSITU_IMGS = imsitu_path('of500_images')
IMSITU_TRAIN_LIST = imsitu_path('train.txt')
IMSITU_VAL_LIST = imsitu_path('val.txt')
IMSITU_TEST_LIST = imsitu_path('test.txt')
IMSITU_VERBS = path('imsitu_verbs.txt')

GLOVE = path('glove.6B.300d')
WORD2VEC = path('glove.6B.300d')

CHECKPOINT_PATH = os.path.join(ROOT_PATH, 'checkpoints')



class ModelConfig(object):
    """Wrapper class for model hyperparameters."""
    def __init__(self):
        """
        Defaults
        """
        self.roi_vision = None
        self.global_vision = None
        self.glove_init = None
        self.spatial_feats = None
        self.ckpt = None
        self.save_dir = None
        self.lr = None
        self.eps = None
        self.sampler = None
        self.beta1 = None
        self.beta2 = None


        self.parser = self.setup_parser()
        self.args = vars(self.parser.parse_args())

        print("~~~~~~~~ Hyperparameters used: ~~~~~~~")
        for x, y in self.args.items():
            print("{} : {}".format(x, y))

        self.__dict__.update(self.args)

        self.save_dir = os.path.join(ROOT_PATH, 'checkpoints', self.save_dir)
        self.ckpt = os.path.join(ROOT_PATH, 'checkpoints', self.ckpt)

        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        assert self.val_size >= 0

        if len(self.ckpt) > 0 and not os.path.exists(self.ckpt):
            raise ValueError("Ckpt file ({}) doesnt exist".format(self.ckpt))

    def setup_parser(self):
        """
        Sets up an argument parser
        :return:
        """
        parser = ArgumentParser(description='training code')
        parser.add_argument('-roi_vision', dest='roi_vision', help='use visual features for the ROIs', action='store_true')
        parser.add_argument('-global_vision', dest='global_vision', help='use global visual features', action='store_true')
        parser.add_argument('-glove_init', dest='glove_init', help='Initialize the word embeddings with GLOVE', action='store_true')
        parser.add_argument('-spatial_feats', dest='spatial_feats', help='Use spatial features also', action='store_true')
        parser.add_argument('-ckpt', dest='ckpt', help='Filename to load from', type=str,
                            default='')
        parser.add_argument('-save_dir', dest='save_dir', help='Directory to save things to, default to checkpoints/save', default='save')
        parser.add_argument('-lr', dest='lr', help='learning rate', type=float, default=1e-3)
        parser.add_argument('-eps', dest='eps', help='Epsilon, for adam', type=float, default=1e-8)
        parser.add_argument('-sampler', dest='sampler', help='Sampler \in \{sg_cls_sampler, ...\}',type=str, default='sg_cls_sampler')
        parser.add_argument('-b', dest='batch_size', help='batch size, default 256',type=int, default=256)
        parser.add_argument('-val_size', dest='val_size', help='val size to use (if 0 we wont use val)', type=int, default=0)
        parser.add_argument('-beta1', dest='beta1', help='for adam', type=float, default=0.9)
        parser.add_argument('-beta2', dest='beta2', help='for adam. probably dont need to touch this', type=float, default=0.999)
        return parser
