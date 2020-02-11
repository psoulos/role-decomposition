from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import torch
from torch.autograd import Variable


use_cuda = torch.cuda.is_available()

# Based on seq2seq tutorial:
# http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

in_header = 'IN:'  # denotes natural language input
out_header = 'OUT:'  # denotes sequence of actions

SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS
        self.max_length = -1

    def addSentence(self, sentence):
        word_list = sentence.split(' ')
        for word in word_list:
            self.addWord(word)
        if len(word_list) > self.max_length:
            self.max_length = len(word_list)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def cut_header(mystr,header):
    # if mystr begins with header, cut it
    n = len(header)
    if mystr[:n] == header:
     return mystr[n:]
    return mystr

def readLangs(fn_in):

    # Read the file and split into lines
    if isinstance(fn_in, str):
        lines = open(fn_in, encoding='utf-8').read().strip().split('\n')
    elif all(isinstance(item, str) for item in fn_in):
        lines = []
        for myfile in fn_in:
            lines += open(myfile, encoding='utf-8').read().strip().split('\n')
    else:
        raise TypeError

    lines = [cut_header(l,in_header) for l in lines]

    # s = unicodeToAscii(s.lower().strip())
    # Split every line into pairs
    pairs = [[s.strip() for s in l.split(out_header)] for l in lines]

    input_lang = Lang('Input')
    output_lang = Lang('Output')
    return input_lang, output_lang, pairs

def prepareData(fn_in, verbose=True):
    # Input
    #  fn_in : input file name (or list of file names to concat.)
    #
    #  Read text file and split into lines, split lines into pairs
    #  Make word lists from sentences in pairs
    if verbose: print("Processing input data")
    if verbose: print(" Reading lines...")
    input_lang, output_lang, pairs = readLangs(fn_in)
    if verbose:
        print(" Read %s sentence pairs" % len(pairs))
        print(" Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    if verbose:
        print(" Counted words:")
        print(' ',input_lang.name, input_lang.n_words)
        print(' ',output_lang.name, output_lang.n_words)
        print('')
    return input_lang, output_lang, pairs

# To train, for each pair we will need an input tensor (indexes of the
# words in the input sentence) and target tensor (indexes of the words in
# the target sentence). While creating these vectors we will append the
# EOS token to both sequences.
def variablesFromPair(pair, input_lang, output_lang):
    # convert sentence pair into indices
    # Input
    #   pair: touple with input and output sentence
    #   input_lang, output_lang : language classes
    input_variable = variableFromSentence(input_lang, pair[0])
    target_variable = variableFromSentence(output_lang, pair[1])
    return (input_variable, target_variable)

def variableFromSentence(lang, sentence):
    # wrap index vector as a variable
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result

def indexesFromSentence(lang, sentence):
    # convert setence string into list of indices
    return [lang.word2index[word] for word in sentence.split(' ')]
