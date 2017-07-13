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
from gensim.models import KeyedVectors

WORD2VEC_PATH = '/home/rowan/code/video-attributes/data/GoogleNews-vectors-negative300.bin.gz'
from unidecode import unidecode

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
    model = KeyedVectors.load_word2vec_format(WORD2VEC_PATH, binary=True)
    wordVectors = {unidecode(w): normalize(model[w]) for w in model.vocab}
    sys.stderr.write("Vectors read from: " + WORD2VEC_PATH + " \n")
    return wordVectors


''' Write word vectors to file '''


def print_word_vecs(wordVectors, outFileName):
    sys.stderr.write('\nWriting down the vectors in ' + outFileName + '\n')
    outFile = open(outFileName, 'w')

    verbs = pd.read_csv('/home/rowan/code/video-attributes/data/attributes.csv')['verb'].unique()
    for word in verbs:
        outFile.write(word + ' ')
        if word not in wordVectors:
            wordVectors[word] = normalize(numpy.random.randn(300))
        for val in wordVectors[word]:
            outFile.write('%.4f' % (val) + ' ')
        outFile.write('\n')
    outFile.close()


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
