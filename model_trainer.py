from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

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

import argparse

from tasks import *
from training import *
from models import *
from evaluation import *
from role_assignment_functions import *

# Code for training a seq2seq RNN on a digit transformation task

parser = argparse.ArgumentParser()
parser.add_argument("--prefix", help="prefix for your training/dev data", type=str, default="digits")
parser.add_argument("--encoder", help="encoder type", type=str, default="ltr")
parser.add_argument("--decoder", help="decoder type", type=str, default="ltr")
parser.add_argument("--task", help="training task", type=str, default="auto")
parser.add_argument("--vocab_size", help="vocab size for the training language", type=int, default=10)
parser.add_argument("--emb_size", help="embedding size", type=int, default=10)
parser.add_argument("--hidden_size", help="hidden size", type=int, default=60)
parser.add_argument("--generalization_prefix", help="prefix for generalization test set", type=str, default=None)
parser.add_argument("--initial_lr", help="initial learning rate", type=float, default=0.001)
parser.add_argument("--batch_size", help="batch size", type=int, default=32)
parser.add_argument("--train", help="whether to train the model or not", type=str, default="True")
parser.add_argument("--file_prefix", help="prefix of file to load and evaluate on", type=str, default=None)
parser.add_argument("--prefix_prefix", help="start of the file prefix", type=str, default="")
parser.add_argument("--joint", help="jointly train with a tpr", type=str, default="False")
parser.add_argument("--direct", help="whether to have direct error term for joint training", type=str, default="False")
parser.add_argument("--role_scheme", help="role scheme for joint tpr", type=str, default="ltr")
parser.add_argument("--role_embeddings", help="file for role embeddings to download", type=str, default=None)
parser.add_argument("--filler_embeddings", help="file for filler embeddings to download", type=str, default=None)
parser.add_argument("--filler_dim", help="filler dimension for joint tpr", type=int, default=10)
parser.add_argument("--role_dim", help="role dimension for joint tpr", type=int, default=10)
parser.add_argument("--n_fillers", help="number of fillers for joint tpr", type=int, default=10)
parser.add_argument("--gen_tasks", help="task for generalization set", type=str, nargs="*")
parser.add_argument("--patience", help="number of epochs to wait", type=int, default=1)
parser.add_argument("--max_length", help="maximum length of a sequence", type=int, default=10)
parser.add_argument(
    "--output_dir",
    help="An optional output folder where files can be saved to.",
    type=str,
    default=None
)
args = parser.parse_args()

output_dir = None
if args.output_dir:
    output_dir = os.path.join('output/model_trainer', args.output_dir)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, 'arguments.txt'), 'w') as arguments_file:
        for argument, value in vars(args).items():
            arguments_file.write('{}: {}\n'.format(argument, value))

use_cuda = torch.cuda.is_available()
if use_cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Load the data sets
with open('data/' + args.prefix + '.train.pkl', 'rb') as handle:
    train_set = pickle.load(handle)

with open('data/' + args.prefix + '.dev.pkl', 'rb') as handle:
    dev_set = pickle.load(handle)

with open('data/' + args.prefix + '.test.pkl', 'rb') as handle:
    test_set = pickle.load(handle)

if args.generalization_prefix is not None:
    with open('data/' + args.generalization_prefix + '.test.pkl', 'rb') as handle:
        generalization_set = pickle.load(handle)

# Define the training function
input_to_output = lambda sequence: transform(sequence, args.task)


# Define the architecture
if args.encoder == "ltr":
    encoder = EncoderRNN(args.vocab_size, args.emb_size, args.hidden_size)
elif args.encoder == "bi":
    encoder = EncoderBiRNN(args.vocab_size, args.emb_size, args.hidden_size)
elif args.encoder == "tree":
    encoder = EncoderTreeRNN(args.vocab_size, args.emb_size, args.hidden_size)
else:
    print("Invalid encoder type")

if args.decoder == "ltr":
    decoder = DecoderRNN(args.vocab_size, args.emb_size, args.hidden_size)
elif args.decoder == "bi":
    decoder = DecoderBiRNN(args.vocab_size, args.emb_size, args.hidden_size)
elif args.decoder == "tree":
    decoder = DecoderTreeRNN(args.vocab_size, args.emb_size, args.hidden_size)
else:
    print("Invalid decoder type")

if use_cuda:
    encoder = encoder.cuda()
    decoder = decoder.cuda()

# Set the prefix for saving weights
file_prefix = args.prefix_prefix + "_" + args.encoder + "_" + args.decoder + "_" + args.task + "_"
directories = os.listdir("./models")
found = False
suffix = 0
while not found:
    if "encoder_" + file_prefix + str(suffix) + ".weights" not in directories:
        found = 1
    else:
        suffix += 1

suffix = str(suffix)

# Train the model
if args.train == "True" and args.joint == "False":
    file_prefix = file_prefix + suffix
    if output_dir:
        encoder_file = os.path.join(output_dir,
                                    'tpr_encoder_{}.weights'.format(file_prefix))
        decoder_file = os.path.join(output_dir,
                                    'decoder_{}.weights'.format(file_prefix))
    else:
        encoder_file = "models/encoder_" + file_prefix + ".weights"
        decoder_file = "models/decoder_" + file_prefix + ".weights"

    train_iters(encoder, decoder, train_set, dev_set, file_prefix, input_to_output,
                max_epochs=200, patience=args.patience, print_every=10000//32, learning_rate=0.001,
                batch_size=args.batch_size)
elif args.train == "True" and args.joint == "True":
        pass # To be added
elif args.train == "True" and args.joint == "Mix":
        file_prefix = file_prefix + suffix

        if output_dir:
            encoder_file = os.path.join(output_dir,
                                        'tpr_encoder_{}.weights'.format(file_prefix))
            decoder_file = os.path.join(output_dir,
                                        'decoder_{}.weights'.format(file_prefix))
        else:
            encoder_file = "models/tpr_encoder_" + file_prefix + ".weights"
            decoder_file = "models/decoder_" + file_prefix + ".weights"

        if args.role_scheme == "bow":
            n_r, seq_to_roles = create_bow_roles(args.max_length, args.n_fillers)
        elif args.role_scheme == "ltr":
            n_r, seq_to_roles = create_ltr_roles(args.max_length, args.n_fillers)
        elif args.role_scheme == "rtl":
            n_r, seq_to_roles = create_rtl_roles(args.max_length, args.n_fillers)
        elif args.role_scheme == "bi":
            n_r, seq_to_roles = create_bidirectional_roles(args.max_length, args.n_fillers)
        elif args.role_scheme == "wickel":
            n_r, seq_to_roles = create_wickel_roles(args.max_length, args.n_fillers)
        elif args.role_scheme == "tree":
            n_r, seq_to_roles = create_tree_roles(args.max_length, args.n_fillers)
        else:
            print("Invalid role scheme")

        tpr_encoder = TensorProductEncoder(
            n_roles=n_r,
            n_fillers=args.n_fillers,
            final_layer_width=args.hidden_size,
            filler_dim=args.filler_dim,
            role_dim=args.role_dim)
        if use_cuda:
            tpr_encoder = tpr_encoder.cuda()

        train_iters_mix(encoder, decoder, tpr_encoder, seq_to_roles, train_set, dev_set,
                        file_prefix, input_to_output, encoder_file=encoder_file,
                        decoder_file=decoder_file, max_epochs=200, patience=args.patience,
                        print_every=10000//32, learning_rate=0.001, batch_size=args.batch_size,
                        output_dir=output_dir)
else:
    file_prefix = args.file_prefix


if args.joint == "Mix":
    # Evaluate the trained model
    tpr_encoder.load_state_dict(torch.load(encoder_file, map_location=device))
    decoder.load_state_dict(torch.load(decoder_file, map_location=device))

    if output_dir:
        report_file = open(os.path.join(output_dir, 'results.txt'), "w")
    else:
        report_file = open("models/results_" + file_prefix + ".txt", "w")

    filler_dict = {}
    for i in range(args.n_fillers):
        filler_dict[i] = i

    correct, total = score3(tpr_encoder, decoder, input_to_output, batchify(test_set, 1), filler_dict, seq_to_roles)
    report_file.write("Test set results:\nCorrect:\t" + str(correct) + "\nTotal:\t" + str(total) + "\nAccuracy:\t" + str(correct * 1.0 / total) + "\n\n")


    if args.generalization_prefix is not None:

        for task in args.gen_tasks:
                input_to_output = lambda sequence: transform(sequence, task)
                correct, total = score3(tpr_encoder, decoder, input_to_output, batchify(generalization_set, 1), filler_dict, seq_to_roles)
                report_file.write("Generalization set results:" + task + "\nCorrect:\t" + str(correct) + "\nTotal:\t" + str(total) + "\nAccuracy:\t" + str(correct * 1.0 / total) + "\n\n")
else:
    # Evaluate the trained model
    encoder.load_state_dict(torch.load(encoder_file, map_location=device))
    decoder.load_state_dict(torch.load(decoder_file, map_location=device))

    if output_dir:
        report_file = open(os.path.join(output_dir, 'results.txt', "w"))
    else:
        report_file = open("models/results_" + file_prefix + ".txt", "w")

    correct, total = score(encoder, decoder, batchify(test_set, 1), input_to_output)
    report_file.write("Test set results:\nCorrect:\t" + str(correct) + "\nTotal:\t" + str(total) + "\nAccuracy:\t" + str(correct * 1.0 / total) + "\n\n")

    if args.generalization_prefix is not None:
        correct, total = score(encoder, decoder, batchify(generalization_set, 1), input_to_output)
        report_file.write("Generalization set results:\nCorrect:\t" + str(correct) + "\nTotal:\t" + str(total) + "\nAccuracy:\t" + str(correct * 1.0 / total) + "\n\n")






