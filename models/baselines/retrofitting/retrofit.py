"""
Contains code to retrofit the GLOVE vectors
https://github.com/mfaruqui/retrofitting
"""

import argparse
import gzip
import math
import numpy
import re
import sys
import pandas as pd

from copy import deepcopy
from text.torchtext.vocab import load_word_vectors
from data.attribute_loader import _load_attributes
from config import GLOVE_PATH, GLOVE_TYPE
from unidecode import unidecode
import pickle as pkl
isNumber = re.compile(r'\d+.*')


def norm_word(word):
    if isNumber.search(word.lower()):
        return '---num---'
    elif re.sub(r'\W+', '', word) == '':
        return '---punc---'
    else:
        return word.lower()


''' Read all the word vectors and normalize them '''


def normalize(x):
    return x / numpy.sqrt(numpy.square(x).sum() + 1.0e-6)


def read_word_vecs():
    wv_dict, wv_arr, _ = load_word_vectors(GLOVE_PATH, GLOVE_TYPE, 300)
    wv_arr = wv_arr.numpy()
    wordVectors = {unidecode(w):normalize(wv_arr[ind]) for w,ind in wv_dict.items()}
    sys.stderr.write("Vectors read from: " + GLOVE_PATH + " \n")
    return wordVectors


''' Write word vectors to file '''
def print_word_vecs(wordVectors, outFileName):
    sys.stderr.write('\nWriting down the vectors in ' + outFileName + '\n')
    # outFile = open(outFileName, 'w')

    verbs = _load_attributes(imsitu_only=False)['template'].unique()
    vecs = {}
    for word in verbs:
        if word in wordVectors:
            vecs[word] = wordVectors[word]
        elif len(word.split(' ')) > 1:
            t0, t1 = word.split(' ')
            vecs[word] = (wordVectors[t0] + wordVectors[t1])/2.0
        elif word.startswith('un'):
            vecs[word] = (wordVectors['un'] + wordVectors[word[2:]])/2.0

    with open(outFileName,'wb') as f:
        pkl.dump(vecs, f)


''' Read the PPDB word relations as a dictionary '''
def read_lexicon(filename, wordVecs):
    lexicon = {}
    for line in open(filename, 'r'):
        words = line.lower().strip().split()
        lexicon[norm_word(words[0])] = [norm_word(word) for word in words[1:]]
    return lexicon


''' Retrofit word vectors to a lexicon '''
def retrofit(wordVecs, lexicon, numIters):
    newWordVecs = deepcopy(wordVecs)
    wvVocab = set(newWordVecs.keys())
    loopVocab = wvVocab.intersection(set(lexicon.keys()))
    for it in range(numIters):
        # loop through every node also in ontology (else just use data estimate)
        for word in loopVocab:
            wordNeighbours = set(lexicon[word]).intersection(wvVocab)
            numNeighbours = len(wordNeighbours)
            # no neighbours, pass - use data estimate
            if numNeighbours == 0:
                continue
            # the weight of the data estimate if the number of neighbours
            newVec = numNeighbours * wordVecs[word]
            # loop over neighbours and add to new vector (currently with weight 1)
            for ppWord in wordNeighbours:
                newVec += newWordVecs[ppWord]
            newWordVecs[word] = newVec / (2 * numNeighbours)
    return newWordVecs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("-i", "--input", type=str, default=None, help="Input word vecs")
    parser.add_argument("-l", "--lexicon", type=str, default=None, help="Lexicon file name")
    parser.add_argument("-o", "--output", type=str, help="Output word vecs")
    parser.add_argument("-n", "--numiter", type=int, default=10, help="Num iterations")
    args = parser.parse_args()

    wordVecs = read_word_vecs()
    lexicon = read_lexicon(args.lexicon, wordVecs)
    numIter = int(args.numiter)
    outFileName = args.output

    ''' Enrich the word vectors using ppdb and print the enriched vectors '''
    print_word_vecs(retrofit(wordVecs, lexicon, numIter), outFileName)
