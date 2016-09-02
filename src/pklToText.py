#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np
import h5py
import itertools
from collections import defaultdict
import pickle
import sys
import re
import codecs
from itertools import izip

pklDlgFileToConvert = "../data/MovieTriples/Training.triples.pkl"
pklVocabDictFile = "../data/MovieTriples/Training.dict.pkl"
textOutputFile = "../data/MovieTriples/VikramTraining.triple.text"
word2VecFile = "../data/MovieTriples/Word2Vec_WordEmb.pkl"

def convertPklToTxt(aRgs):

    textOutputFileHandler = open(textOutputFile,"w")

    # load the vocabulary
    print "Loading the vocab file....."
    pklVocabDictFileHandler = open(pklVocabDictFile,"r")
    pklVocabDictFileContent = pickle.load(pklVocabDictFileHandler)
    print "Populating the vocab hash map...."

    indices_to_word = {}
    word_to_indices = {}
    for word_ex in pklVocabDictFileContent:
        indices_to_word[word_ex[1]] = word_ex[0]
        word_to_indices[word_ex[0]] = word_ex[1]

    print indices_to_word[3]
    print indices_to_word[4]
    print indices_to_word[10001]
    print indices_to_word[10002]

    print word_to_indices["<s>"]
    print word_to_indices["</s>"]
    print word_to_indices["<unk>"]

    #load the conversations
    print "Loading the dialog file....."
    pklDlgFileHandler = open(pklDlgFileToConvert,"r")
    pklDlgFileContent = pickle.load(pklDlgFileHandler)
    print "Loaded the dialog file.....iterating through it..."+str(len(pklDlgFileContent))

    maxCnt = 196308
    for j in range(len(pklDlgFileContent)):
        if(j>maxCnt):
            break
        lineInIndices = pklDlgFileContent[j]
        lineInWords = ""
        for k in range(len(lineInIndices)):
            lineInWords=lineInWords+" "+indices_to_word[lineInIndices[k]]
        textOutputFileHandler.write(lineInWords)
        textOutputFileHandler.write("\n")
    print "Done!!"

def main(arguments):

    word2VecFileHandler = open(word2VecFile,"r")
    word2VecFileContent = pickle.load(word2VecFileHandler)

    convertPklToTxt(arguments)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
