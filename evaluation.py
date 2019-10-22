from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
from random import shuffle

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import sys
import os

import time
import math

import pickle


import data_loader as data
from role_assignment_functions import *
from rolelearner.role_learning_tensor_product_encoder import RoleLearningTensorProductEncoder

# Functions for evaluating seq2seq models and TPDNs

use_cuda = torch.cuda.is_available()
if use_cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Given an encoder, a decoder, and an input, return the guessed output
# and the encoding of the input
def evaluate(encoder1, decoder1, example, input_to_output):
    encoding = encoder1([example])
    predictions = decoder1(encoding, len(example), [parse_digits(example)])
    correct = input_to_output(example)
        
    guessed_seq = []
        
    for prediction in predictions:
        topv, topi = prediction.data.topk(1)
        ni = topi.item() 
            
        guessed_seq.append(ni)
        
    return guessed_seq, encoding
            
# Given an encoder, decoder, evaluation set, and function for generating
# the correct outputs, return the number of correct predictions
# and the total number of predictions
def score(encoder1, decoder1, evaluation_set, input_to_output):
    total_correct = 0
    total = 0
    

    for batch in evaluation_set:
            for example in batch:
                correct = input_to_output(example)
        
                guess = evaluate(encoder1, decoder1, example, input_to_output)
        
                if tuple(guess[0]) == tuple(correct):
                    total_correct += 1
                total += 1
        
    return total_correct, total

# This function takes a tensor product encoder and a standard decoder, as well as a sequence
# of digits as inputs. It then uses the tensor product encoder to encode the sequence and uses
# the standard decoder to decode it, and returns the result.
def evaluate2(encoder, decoder, example):
   
    if use_cuda:
        if isinstance(encoder, RoleLearningTensorProductEncoder):
            encoder_hidden = encoder(
                Variable(torch.LongTensor(example[0])).cuda().unsqueeze(0),
                Variable(torch.LongTensor(example[1])).cuda().unsqueeze(0)
            )[0]
        else:
            encoder_hidden = encoder(Variable(torch.LongTensor(example[0])).cuda().unsqueeze(0), Variable(torch.LongTensor(example[1])).cuda().unsqueeze(0))
    else:
        if isinstance(encoder, RoleLearningTensorProductEncoder):
            encoder_hidden = encoder(
                Variable(torch.LongTensor(example[0])).unsqueeze(0),
                Variable(torch.LongTensor(example[1])).unsqueeze(0)
            )[0]
        else:
            encoder_hidden = encoder(Variable(torch.LongTensor(example[0])).unsqueeze(0), Variable(torch.LongTensor(example[1])).unsqueeze(0))
    predictions = decoder(encoder_hidden, len(example[0]), [parse_digits(example[0])])
        
    guessed_seq = []
    for prediction in predictions:
        topv, topi = prediction.data.topk(1)
        ni = topi.item()
            
        guessed_seq.append(ni)
    
        
    return guessed_seq


# This function takes a tensor product encoder and a standard decoder, as well as a sequence
# of digits as inputs. It then uses the tensor product encoder to encode the sequence and uses
# the standard decoder to decode it, and returns the result.
def evaluate3(encoder, decoder, example, role_function):
    #print("EXAMPLE:", example)  

    if use_cuda:
        encoder_hidden = encoder(Variable(torch.LongTensor(example)).cuda().unsqueeze(0), Variable(torch.LongTensor(role_function(example))).cuda().unsqueeze(0))
    else:
        encoder_hidden = encoder(Variable(torch.LongTensor(example)).unsqueeze(0), Variable(torch.LongTensor(role_function(example))).unsqueeze(0))
    predictions = decoder(encoder_hidden, len(example), [parse_digits(example)])

    guessed_seq = []
    for prediction in predictions:
        topv, topi = prediction.data.topk(1)
        ni = topi.item()

        guessed_seq.append(ni)


    return guessed_seq

# This function takes a tensor product encoder and a standard decoder, as well as a sequence
# of digits as inputs. It then uses the tensor product encoder to encode the sequence and uses
# the standard decoder to decode it, and returns the result.
def evaluate4(encoder, decoder, example, role_function):
    #print("EXAMPLE:", example)  

    if use_cuda:
        encoder_hidden = encoder(Variable(torch.LongTensor(example)).cuda().unsqueeze(0), Variable(torch.LongTensor(role_function(example))).cuda().unsqueeze(0))
    else:
        encoder_hidden = encoder(Variable(torch.LongTensor(example)).unsqueeze(0), Variable(torch.LongTensor(role_function(example))).unsqueeze(0))
    predictions = decoder(encoder_hidden, len(example), [parse_digits(example)])

    guessed_seq = []
    for prediction in predictions:
        topv, topi = prediction.data.topk(1)
        ni = topi.item()

        guessed_seq.append(ni)


    return guessed_seq, encoder_hidden

def score2(encoder, decoder, input_to_output, test_set, index_to_filler):
    # Evaluate this TPR encoder for how well it can encode sequences in a way
    # that our original mystery_decoder can decode
    accurate = 0
    total = 0

    for batch in test_set:
        for example in batch:
            example = example[0]
            pred = evaluate2(encoder, decoder, example)
            
                     
            if tuple(input_to_output([index_to_filler[x] for x in example[0]])) == tuple([str(x) for x in pred]):
                accurate += 1
            total += 1
    
    # Gives how many sequences were properly decoded, out of the total number of test sequences    
    return accurate, total

def scoreSCAN(encoder, decoder, input_to_output, test_set, index_to_filler, output_lang,
              feed_input=True):
    accurate = 0
    total = 0

    max_length = output_lang.max_length

    for batch in test_set:
        for example in batch:
            example = example[0]

            sequence = Variable(torch.LongTensor(example[0])).unsqueeze(0)
            sequence = sequence.cuda() if use_cuda else sequence
            true_roles = Variable(torch.LongTensor(example[1])).unsqueeze(0)
            true_roles = true_roles.cuda() if use_cuda else true_roles

            if isinstance(encoder, RoleLearningTensorProductEncoder):
                encoder_hidden = encoder(sequence, true_roles)[0]
            else:
                encoder_hidden = encoder(sequence, true_roles)

            decoder_input = Variable(torch.LongTensor([[data.SOS_token]], device=device))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            # (num_layers * num_directions, batch, hidden_size)
            decoder_hidden = encoder_hidden.view((1, 1, -1))
            # use last state of encoder to start the decoder

            decoded_words = []
            for di in range(max_length):
                if not feed_input:
                    decoder_input = torch.zeros_like(decoder_input)
                decoder_output, decoder_hidden, decoder_raw_score = decoder(decoder_input,
                                                                            decoder_hidden)

                topv, topi = decoder_output.data.topk(2)
                ni = topi[0][0]

                # the network terminates the string
                if ni == data.EOS_token:
                    break

                decoded_words.append(output_lang.index2word[ni.item()])
                decoder_input = Variable(torch.LongTensor([[ni]]))
                decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            true_output = input_to_output(' '.join([index_to_filler[x] for x in example[0]]))
            decoded_words = ' '.join(decoded_words)
            if true_output == decoded_words:
                accurate += 1
            total += 1

    return accurate, total


def score3(encoder, decoder, input_to_output, test_set, index_to_filler, role_function):
    # Evaluate this TPR encoder for how well it can encode sequences in a way
    # that our original mystery_decoder can decode
    accurate = 0
    total = 0

    for batch in test_set:
        for example in batch:
            #print("FIRST EXAMPLE", example)
            #example = example[0]
            pred = evaluate3(encoder, decoder, example, role_function)

            #print(example, pred)         
            if tuple(input_to_output([str(index_to_filler[x]) for x in example])) == tuple([str(x) for x in pred]):
                accurate += 1
            else:
                print(example, pred)
            total += 1

    # Gives how many sequences were properly decoded, out of the total number of test sequences    
    return accurate, total
