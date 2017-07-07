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
    return os.path.join(IMSITU_PATH, fn)



ATTRIBUTES_PATH = vwa_path('attributes.csv')
ATTRIBUTES_SPLIT = vwa_path('attributes_split.csv')
DEFNS_PATH = vwa_path('verb_definitions.csv')
DICTIONARY_PATH = path('dictionary_challenge.pkl')

IMSITU_LABELS = imsitu_path('OpenFrame500.tab')
IMSITU_IMGS = imsitu_path('of500_images')
IMSITU_TRAIN_LIST = imsitu_path('train_set.txt')
IMSITU_VAL_LIST = imsitu_path('dev_set.txt')
IMSITU_TEST_LIST = imsitu_path('test_set.txt')
IMSITU_VERBS = path('imsitu_verbs.txt')

GLOVE = path('glove.6B.300d')
WORD2VEC = path('glove.6B.300d')

CHECKPOINT_PATH = os.path.join(ROOT_PATH, 'checkpoints')
INIT_SCALE= 2e-5


class ModelConfig(object):
    """Wrapper class for model hyperparameters."""
    def __init__(self, margin=0.1, lr=1e-3, batch_size=64, eps=1e-8, beta1=0.9, beta2=0.999,
                 ckpt='', save_dir='save'):
        """
        Defaults
        """
        self.margin = margin
        self.lr = lr
        self.batch_size = batch_size
        self.eps = eps
        self.beta1 = beta1
        self.beta2 = beta2
        self.ckpt = ckpt
        self.save_dir = save_dir

        self.parser = self.setup_parser()
        self.args = vars(self.parser.parse_args())

        print("~~~~~~~~ Hyperparameters used: ~~~~~~~")
        for x,y in self.args.items():
            val = self.__dict__[x] if y is None else y
            print("{} : {}".format(x, val))

        self.__dict__.update({x:y for x, y in self.args.items() if y is not None})

        self.save_dir = os.path.join(ROOT_PATH, 'checkpoints', self.save_dir)

        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        if len(self.ckpt) > 0:
            self.ckpt = os.path.join(ROOT_PATH, 'checkpoints', self.ckpt)
            if not os.path.exists(self.ckpt):
                raise ValueError("Ckpt file ({}) doesnt exist".format(self.ckpt))

    def setup_parser(self):
        """
        Sets up an argument parser
        :return:
        """
        parser = ArgumentParser(description='training code')
        parser.add_argument('-ckpt', dest='ckpt', help='Filename to load from', type=str)
        parser.add_argument('-save_dir', dest='save_dir', help='Directory to save things to')
        parser.add_argument('-lr', dest='lr', help='learning rate', type=float)
        parser.add_argument('-eps', dest='eps', help='Epsilon, for adam', type=float)
        parser.add_argument('-b', dest='batch_size', help='batch size', type=int)
        parser.add_argument('-beta1', dest='beta1', help='for adam', type=float,)
        parser.add_argument('-beta2', dest='beta2', help='for adam. probably dont need to touch this', type=float)
        return parser
