#!/usr/bin/env python

import argparse
import random
import hashlib
import shelve
import time
import logging

import numpy as np

import chainer
from chainer import cuda
from chainer import optimizers
from datasets import Indexer

from gensim.models import KeyedVectors
from model import RecursiveNet
from sklearn.decomposition import PCA


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--epoch', '-e', default=100, type=int,
                    help='number of epochs.')
parser.add_argument('--unit', '-u', default=100, type=int,
                    help='number of units')
parser.add_argument('--batchsize', '-b', type=int, default=20,
                    help='learning minibatch size')
parser.add_argument('--epocheval', '-p', type=int, default=5,
                    help='number of epochs per evaluation')
parser.add_argument('--grammar-filename', '-G', type=str,
                    help='grammar object by running dump_grammar.py')
parser.add_argument('--learning-rate', '-l', type=float, default=0.01,
                    help='Learning rate value.')
parser.add_argument('--l2_reg', '-L', type=float, default=0.01,
                    help='L2 regularization coeff.')
parser.add_argument('--logging', type=str, default='clm.log',
                    help='Logging file.')
parser.add_argument('--test', dest='test', action='store_true')
parser.add_argument('--em-epoch', '-E', default=10, type=int,
                    help='number of EM epochs.')
parser.set_defaults(test=False)

args = parser.parse_args()

logging.basicConfig(filename=args.logging, level=logging.INFO)

if args.gpu >= 0:
    cuda.check_cuda_available()

xp = cuda.cupy if args.gpu >= 0 else np

n_epoch = args.epoch        # number of epochs.
n_units = args.unit         # number of units per layer.
batchsize = args.batchsize  # minibatch size.
epoch_per_eval = args.epocheval  # number of epochs per evaluation.
learning_rate = args.learning_rate
l2_reg = args.l2_reg
n_em_epoch = args.em_epoch  # number of EM epoch.

#chainer.set_debug(True)

# Get preprocessed Grammar
grammar = shelve.open(args.grammar_filename) \
            if args.grammar_filename else None

dataset_train = './data/train1'
dataset_valid = '../data/test'
dataset_test = '../data/test'
word2vec_file = '~/GoogleNews-vectors-negative300.bin'

indexer = Indexer(dataset_train)
indexer.build_vocab()


def generate_rnn_grammar(sentence):
    A = {}
    length = len(indexer.tokenize(sentence))
    for end in range(1, length + 1):
        span = (0, end)
        A[span] = {end - 1 : 1}

    return A

def get_sentence_grammar(sentence):
    if grammar is not None:
        hash = hashlib.md5(sentence).hexdigest()
        return grammar[hash]
    return generate_rnn_grammar(sentence) 

def evaluate(model, test_sents):
    m = model.copy()
    m.clear_z_leaf()
    m.init_z_leaf(words)
    entropy = 0
    for sentence in test_sents:
        sentence = sentence.strip()
        sentence_grammar = get_sentence_grammar(sentence) 
        indexed_sentence = indexer.index(sentence)
        length = len(indexed_sentence)
        A = inside(m, indexed_sentence, length, sentence_grammar, train=False)
        entropy += -np.log2(A[0, length][0]) + length * np.log2(m.z_leaf.data)
    return entropy


def embedding_vectors(n_units, vocab, word2vec_file):
    model = KeyedVectors.load_word2vec_format(word2vec_file, binary=True)
    # return emb_300
    pca = PCA(n_components=n_units)
    return pca.fit_transform(emb_300)

vocab = indexer.get_vocab()
n_vocab = len(vocab)
# initW = embedding_vectors(n_units, vocab, word2vec_file)
initW = xp.random.uniform(-1, 1, (n_vocab, n_units))
model = RecursiveNet(n_vocab, n_units, initW)

words = xp.array(xrange(n_vocab), np.int32)

if args.gpu >= 0:
    model.to_gpu()

# Setup optimizer
optimizer = optimizers.AdaGrad(lr=learning_rate)
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.WeightDecay(l2_reg))

accum_loss = long()
batch_count = 0
start_at = time.time()
cur_at = start_at

logging.info('Vocab size: {}'.format(n_vocab))

train_size = 0
with open(dataset_train) as f:
    train_sentences = f.readlines()
    train_size += len(train_sentences)

test_size = 0
with open(dataset_test) as f:
    test_sentences = f.readlines()
    test_size += len(test_sentences)

valid_size = 0
with open(dataset_valid) as f:
    valid_sentences = f.readlines()
    valid_size += len(valid_sentences)

logging.info('Training dataset size: {}'.format(train_size))
logging.info('Valid dataset size: {}'.format(valid_size))
logging.info('Test dataset size: {}'.format(test_size))

#logging.info('Train data evaluation: %.5f'
#             % evaluate(model, train_sentences))

import pdb;pdb.set_trace()
for epoch in range(n_epoch):
    logging.info('Epoch: {0:d}'.format(epoch))
    epoch_count = 0
    total_loss = long()
    batch = 0
    cur_at = time.time()
    random.shuffle(train_sentences)

    model.init_z_leaf(words)
    As = {}
    Mus = {}
    sentences = {}
    for i, sentence in enumerate(train_sentences):
        sentence = sentence.strip()
        length = len(sentence.split())
        logging.info('Processing sentence#{}:{}::{}'
                     .format(i, length, sentence))

        hash = hashlib.md5(sentence).hexdigest()
        sentence_grammar = get_sentence_grammar(sentence) 
        indexed_sentence = indexer.index(sentence)
        loss = model.forward(xp.array(indexed_sentence, np.int32),
                          sentence_grammar)

        logging.debug('Processed sentence#{}=>{}'
                      .format(i, loss.data))

        epoch_count += 1
        batch_count += 1
        
        accum_los += loss

        if batch_count >= batchsize:
            logging.info('Updating batch gradient for batch: {}'
                         .format(batch))
            batch += 1
            logging.info('loss: {:.5f}'.format(float(accum_loss.data)))
            try:
                accum_loss.backward()
                optimizer.update()
            except Exception, e:
                logging.error("Failed update for batch %d\n%s"
                              % (batch, e))
            finally:
                accum_loss = 0
                model.cleargrads()

            model.clear_z_leaf()
            model.init_z_leaf(words)
            batch_count = 0

    now = time.time()
    throughput = float(train_size) / (now - cur_at)
    logging.info('{:.2f} iters/sec, {:.2f} sec'
                 .format(throughput, now - cur_at))

    if (epoch + 1) % epoch_per_eval == 0:
        logging.info('Train data evaluation: %.5f'
                     % evaluate(model, train_sentences))
        # logging.info('Develop data evaluation: %.5f'
        #              % evaluate(model, valid_sentences))
logging.info('Test evaluateion')
logging.info('Test loss: {:.5f}'
             .format(float(evaluate(model, test_sentences))))
