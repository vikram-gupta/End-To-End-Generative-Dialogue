#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Create the data for the LSTM.
"""

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

class Indexer:
    def __init__(self, symbols = ["*blank*","<unk>","<s>","</s>"]):
        # This needs to be changed so that it loads the proper mappings from word to characters
        self.vocab = defaultdict(int)
        self.PAD = symbols[0]
        self.UNK = symbols[1]
        self.BOS = symbols[2]
        self.EOS = symbols[3]
        self.d = {self.PAD: 1, self.UNK: 2, self.BOS: 3, self.EOS: 4}

    def add_w(self, ws):
        for w in ws:
            if w not in self.d:
                self.d[w] = len(self.d) + 1
            
    def convert(self, w):
        return self.d[w] if w in self.d else self.d[self.UNK]

    def convert_sequence(self, ls):
        return [self.convert(l) for l in ls]

    def clean(self, s):
        s = s.replace(self.PAD, "")
        # s = s.replace(self.UNK, "")
        s = s.replace(self.BOS, "")
        s = s.replace(self.EOS, "")
        return s
        
    def write(self, outfile):
        out = open(outfile, "w")
        items = [(v, k) for k, v in self.d.iteritems()]
        items.sort()
        for v, k in items:
            print >>out, k.encode('utf-8'), v
        out.close()

    # This sorts the vocab according to frequency count and reduces the 
    # vocab to a certain specified amount
    def prune_vocab(self, k):
        vocab_list = [(word, count) for word, count in self.vocab.iteritems()]
        vocab_list.sort(key = lambda x: x[1], reverse=True)
        k = min(k, len(vocab_list))
        self.pruned_vocab = {pair[0]:pair[1] for pair in vocab_list[:k]}
        for word in self.pruned_vocab:
            if word not in self.d:
                self.d[word] = len(self.d) + 1

    def load_vocab(self, vocab_file):
        self.d = {}
        for line in open(vocab_file, 'r'):
            v, k = line.decode("utf-8").strip().split()
            self.d[v] = int(k)
            
def pad(ls, length, symbol):
    if len(ls) >= length:
        return ls[:length]
    return ls + [symbol] * (length -len(ls))
        
def get_data(args):
    src_indexer = Indexer(["<blank>","<unk>","<s>","</s>"])
    target_indexer = Indexer(["<blank>","<unk>","<s>","</s>"])    
    
    def make_vocab(srcfile, targetfile, seqlength, max_word_l=0, chars=0):
        num_sents = 0
        sent_thrown_out = 0
        for _, (src_orig, targ_orig) in \
                enumerate(itertools.izip(open(srcfile,'r'), open(targetfile,'r'))):
            src_orig = src_indexer.clean(src_orig.decode("utf-8").strip())
            targ_orig = target_indexer.clean(targ_orig.decode("utf-8").strip())
            targ = targ_orig.strip().split()
            src = src_orig.strip().split()
            if len(targ) > seqlength or len(src) > seqlength or len(targ) < 1 or len(src) < 1:
                sent_thrown_out = sent_thrown_out + 1
                continue
            num_sents += 1
            for word in targ:                                
                target_indexer.vocab[word] += 1
                
            for word in src:                 
                src_indexer.vocab[word] += 1
        print('Number of sentences thrown out', sent_thrown_out)
        return max_word_l, num_sents
                
    def convert(srcfile, targetfile, batchsize, seqlength, outfile, num_sents,
                max_word_l, max_sent_l=0,chars=0, unkfilter=0):
        
        newseqlength = seqlength + 2 #add 2 for EOS and BOS
        targets = np.zeros((num_sents, newseqlength), dtype=int)
        target_output = np.zeros((num_sents, newseqlength), dtype=int)
        sources = np.zeros((num_sents, newseqlength), dtype=int)
        source_lengths = np.zeros((num_sents,), dtype=int)
        target_lengths = np.zeros((num_sents,), dtype=int)
        dropped = 0
        sent_id = 0
        for _, (src_orig, targ_orig) in \
                enumerate(itertools.izip(open(srcfile,'r'), open(targetfile,'r'))):
            # Loading the sentences
            src_orig = src_indexer.clean(src_orig.decode("utf-8").strip())
            targ_orig = target_indexer.clean(targ_orig.decode("utf-8").strip())
            targ = [target_indexer.BOS] + targ_orig.strip().split() + [target_indexer.EOS]
            src =  [src_indexer.BOS] + src_orig.strip().split() + [src_indexer.EOS]
            max_sent_l = max(len(targ), len(src), max_sent_l)

            # We're bounding the length of a sequence for a file
            if len(targ) > newseqlength or len(src) > newseqlength or len(targ) < 3 or len(src) < 3:
                dropped += 1
                continue                   

            targ = pad(targ, newseqlength+1, target_indexer.PAD)
            for word in targ:
                word = word if word in target_indexer.d else target_indexer.UNK                 
            targ = target_indexer.convert_sequence(targ)
            targ = np.array(targ, dtype=int)

            src = pad(src, newseqlength, src_indexer.PAD)
            src = src_indexer.convert_sequence(src)
            src = np.array(src, dtype=int)
            
            # Drops all unknown characters
            if unkfilter > 0:
                targ_unks = float((targ[:-1] == 2).sum())
                src_unks = float((src == 2).sum())                
                if unkfilter < 1: #unkfilter is a percentage if < 1
                    targ_unks = targ_unks/(len(targ[:-1])-2)
                    src_unks = src_unks/(len(src)-2)
                if targ_unks > unkfilter or src_unks > unkfilter:
                    dropped += 1
                    continue
                
            targets[sent_id] = np.array(targ[:-1],dtype=int)
            #count non padded characters
            target_lengths[sent_id] = (targets[sent_id] != 1).sum()
            target_output[sent_id] = np.array(targ[1:],dtype=int)                    
            sources[sent_id] = np.array(src, dtype=int)
            #count non padded characters
            source_lengths[sent_id] = (sources[sent_id] != 1).sum()

            sent_id += 1
            if sent_id % 100000 == 0:
                print("{}/{} sentences processed".format(sent_id, num_sents))
                if sent_id > 3000000:
                    break

        #break up batches based on source lengths
        source_lengths = source_lengths[:sent_id]
        source_sort = np.argsort(source_lengths)

        sources = sources[source_sort]
        targets = targets[source_sort]
        target_output = target_output[source_sort]
        target_l = target_lengths[source_sort]
        source_l = source_lengths[source_sort]

        curr_l = 0
        l_location = [] #idx where sent length changes
        
        for j,i in enumerate(source_sort):
            if source_lengths[i] > curr_l:
                curr_l = source_lengths[i]
                l_location.append(j+1)
        l_location.append(len(sources))

        #get batch sizes
        curr_idx = 1
        batch_idx = [1]
        nonzeros = []
        batch_l = []
        batch_w = []
        target_l_max = []
        for i in range(len(l_location)-1):
            while curr_idx < l_location[i+1]:
                curr_idx = min(curr_idx + batchsize, l_location[i+1])
                batch_idx.append(curr_idx)
        #batch idx is storing values at which a new batch is starting for the data
        for i in range(len(batch_idx)-1):
            #number of elements in the batch
            batch_l.append(batch_idx[i+1] - batch_idx[i])            
            #length of the batch
            batch_w.append(source_l[batch_idx[i]-1])
            nonzeros.append((target_output[batch_idx[i]-1:batch_idx[i+1]-1] != 1).sum().sum())
            target_l_max.append(max(target_l[batch_idx[i]-1:batch_idx[i+1]-1]))

        # Write output
        f = h5py.File(outfile, "w")
        
        f["source"] = sources
        f["target"] = targets
        f["target_output"] = target_output
        f["target_l"] = np.array(target_l_max, dtype=int)
        f["target_l_all"] = target_l        
        f["batch_l"] = np.array(batch_l, dtype=int)
        f["batch_w"] = np.array(batch_w, dtype=int)
        f["batch_idx"] = np.array(batch_idx[:-1], dtype=int)
        f["target_nonzeros"] = np.array(nonzeros, dtype=int)
        f["source_size"] = np.array([len(src_indexer.d)])
        f["target_size"] = np.array([len(target_indexer.d)])

        print("Saved {} sentences (dropped {} due to length/unk filter)".format(
            len(f["source"]), dropped))
        f.close()                
        return max_sent_l

    print("First pass through data to get vocab...")
    max_word_l, num_sents_train = make_vocab(args.srcfile, args.targetfile,
                                             args.seqlength, 0, 0)
    print("Number of sentences in training: {}".format(num_sents_train))
    max_word_l, num_sents_valid = make_vocab(args.srcvalfile, args.targetvalfile,
                                             args.seqlength, max_word_l, 0)
    print("Number of sentences in valid: {}".format(num_sents_valid))

    #prune and write vocab
    src_indexer.prune_vocab(args.srcvocabsize)
    target_indexer.prune_vocab(args.targetvocabsize)
    if args.srcvocabfile != '':
        # You can try and load vocabulary here for both the source and target files 
        print('Loading pre-specified source vocab from ' + args.srcvocabfile)
        src_indexer.load_vocab(args.srcvocabfile)
    if args.targetvocabfile != '':
        print('Loading pre-specified target vocab from ' + args.targetvocabfile)
        target_indexer.load_vocab(args.targetvocabfile)
        
    src_indexer.write(args.outputfile + ".src.dict")
    target_indexer.write(args.outputfile + ".targ.dict")
    
    print("Source vocab size: Original = {}, Pruned = {}".format(len(src_indexer.vocab), 
                                                          len(src_indexer.d)))
    print("Target vocab size: Original = {}, Pruned = {}".format(len(target_indexer.vocab), 
                                                          len(target_indexer.d)))

    max_sent_l = 0
    max_sent_l = convert(args.srcvalfile, args.targetvalfile, args.batchsize, args.seqlength,
                         args.outputfile + "-val.hdf5", num_sents_valid,
                         max_word_l, max_sent_l, 0, args.unkfilter)
    max_sent_l = convert(args.srcfile, args.targetfile, args.batchsize, args.seqlength,
                         args.outputfile + "-train.hdf5", num_sents_train, max_word_l,
                         max_sent_l, 0, args.unkfilter)
    print("Max sent length: {}".format(max_sent_l))    


def format_data(args):
    '''
        Formats the data so it can be passed into get_data function
    '''

    def load_data():
        ''' 
            Assuming all inputs files are .pkl files 

            Loads all data files into a data_dict thats first layer
            is the name of the data folder that is processed, and the 
            second layer is the variable name of the loaded file. 

            i.e. 

            data_dict['MovieTriples']['train_set'] = pickle.load(...)
        '''
        data_dict = {}
        for data_set in args.input_files:
            data_dict[data_set] = {}
            for var_name, data_file in args.input_files[data_set].iteritems():
                with open('%s%s/%s' % (args.input_directory, data_set, data_file)) as f:
                    data_dict[data_set][var_name] = f.readlines()

        return data_dict

    def write_indicies_to_file(filename, y):
        '''
            Write the contents of y into filename
        '''
        f =  open(filename, 'w')
        for context in y: 
            for ind in context:
                f.write(str(ind) + ' ')
            f.write('\n')
        f.close()

    def write_vocab_to_file(filename):
        ''' 
            Write the vocabulary to filename
        '''
        with open(filename, 'w') as f: 
            for i in range(1, len(indices_to_word)+1):
                f.write(indices_to_word[i] + ' ' + str(i) + '\n')

    def write_words_to_file(filename, indices_dict):
        '''
            Write the examples to files as words, removing special
            indices
        '''
        lst = []
        for context in indices_dict:
            context_words = []
            for ind in context:
                if ind not in special_indices:
                    context_words.append(indices_to_word[ind])
            lst.append(' '.join(context_words))


        f =  open(filename, 'w')
        for context in lst: 
            f.write(str(context) + ' \n')
        f.close()

    # Load in datafiles
    data_dict = load_data()

    max_len_context = 0
    max_len_output = 0 

    data_set_contexts = []
    data_set_outputs = []

    data_sets = data_dict

    for j in range(len(data_sets)):
        data_set = data_sets[j]

        full_context = []
        full_output = []

        for i in range(len(data_set)):

            toks = data_set[i].split("-delimit-")
            context = toks[0]
            output = toks[1]

            # Cap the target and src length at 302 words to make computation simpler, goes up to ~1500
            if len(context) > max_len_context:
                continue
            if len(output) > max_len_output:
                continue

            full_context.append(context)
            full_output.append(output)

        data_set_contexts.append(full_context)
        data_set_outputs.append(full_output)
        
    train_full_context = data_set_contexts[0]
    train_full_output = data_set_outputs[0]
    valid_full_context = data_set_contexts[1]
    valid_full_output = data_set_outputs[1]

    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    write_words_to_file(args.srcfile, train_full_context)
    write_words_to_file(args.srcvalfile, valid_full_context)
    write_words_to_file(args.targetfile, train_full_output)
    write_words_to_file(args.targetvalfile, valid_full_output)
    print('Done formatting the Kiwi data')


def main(arguments):

    data_name = "Jerry"

    parser = argparse.ArgumentParser(
                description=__doc__,
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Preprocess Modifications
    parser.add_argument('--batchsize', help="Size of each minibatch.", type=int, default=64)
    parser.add_argument('--seqlength', help="Maximum sequence length. Sequences longer "
                                               "than this are dropped.", type=int, default=50)
    parser.add_argument('--unkfilter', help="Ignore sentences with too many UNK tokens. "
                                       "Can be an absolute count limit (if > 1) "
                                       "or a proportional limit (0 < unkfilter < 1).",
                                          type = float, default = 0)

    args = parser.parse_args(arguments)


    args.input_directory = "../data/"+data_name+"/input"
    args.output_directory = "../data/"+data_name+"/output"
    args.outputfile = data_name

    #dictionary
    args.srcvocabfile = 'src.dict'
    args.targetvocabfile = 'targ.dict'

    # Vocabularys
    args.srcvocabsize = 10000
    args.targetvocabsize = 10000

    # Output files in words
    args.srcfile ='train_src_words.txt'
    args.targetfile='train_targ_words.txt'

    args.srcvalfile = 'dev_src_words.txt'
    args.targetvalfile = 'dev_targ_words.txt'

    args.input_files = { 'KiwiData' : { 'train_set' : data_name+'Train.txt',
                                           'valid_set' : data_name+'Validation.txt',
                                       }
                        }

    # Append on output directory to the output files
    output_files = ['srcfile', 'targetfile', 'srcvalfile', 'targetvalfile', 'srcfile_ind', 'targetfile_ind', 
                    'srcvalfile_ind', 'targetvalfile_ind', 'outputfile', 'srcvocabfile', 'targetvocabfile']

                    
    for o_f in output_files:
        setattr(args, o_f, args.output_directory + getattr(args, o_f))

    format_data(args)
    get_data(args)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
