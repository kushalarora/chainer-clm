#!/usr/bin/env python
"""
"""

import argparse
import codecs
import collections
import os
import random
import hashlib
import re
import shelve
import time
import logging

import numpy as np

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
from datasets import Indexer


FLOAT_MIN = 1e-30
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--epoch', '-e', default=400, type=int,
                    help='number of epochs to learn')
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
parser.set_defaults(test=False)

args = parser.parse_args()
logging.basicConfig(filename=args.logging,level=logging.INFO)
if args.gpu >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if args.gpu >= 0 else np

n_epoch = args.epoch       # number of epochs
n_units = args.unit        # number of units per layer
batchsize = args.batchsize      # minibatch size
epoch_per_eval = args.epocheval  # number of epochs per evaluation
learning_rate = args.learning_rate
l2_reg = args.l2_reg

chainer.set_debug(True)
# Get preprocessed Grammar
grammar = shelve.open(args.grammar_filename)

class RecursiveNet(chainer.Chain):

    def __init__(self, n_vocab, n_units):
        super(RecursiveNet, self).__init__(
            embed=L.EmbedID(n_vocab, n_units),
            w=L.Linear(n_units * 2, n_units),
            u=L.Linear(n_units, 1),
            h1=L.Linear(n_units, 1),
            h2=L.Linear(n_units, 1))
        self.n_vocab = n_vocab
        self.n_units = n_units
        self.z_leaf = FLOAT_MIN

    def leaf(self, x):
        return self.embed(x)

    def node(self, left, right):
        return F.tanh(self.w(F.concat((left, right))))

    def leaf_unprob(self, v):
        return F.exp(self.u(v))

    def comp_unprob(self, parent, left, right):
        return F.exp(self.u(parent) + self.h1(left) + self.h2(right))

    def init_z_leaf(self, train=False):
        for i in xrange(0, self.n_vocab):
            word = xp.array([i], np.int32)
            x = chainer.Variable(word, volatile=not train)
            self.z_leaf += self.leaf_unprob(self.leaf(x))

    def clear_z_leaf(self):
        self.z_leaf = FLOAT_MIN

def traverse(model, sent, length, sentence_grammar, train=True):
    assert isinstance(sent, list)

    Z = {}
    node_map = {}

    # leaf nodes
    for index in xrange(0, length):
        span = (index, index + 1)
        split = index

        word = xp.array([sent[index]], np.int32)
        x = chainer.Variable(word, volatile=not train)
        node = model.leaf(x)
        node_map[span] = node
        Z[span] = model.leaf_unprob(node)

    # internal nodes
    for diff in xrange(2, length + 1):
        for start in xrange(0, length - diff + 1):
            end = start + diff
            span = (start, end)
            comp_z = FLOAT_MIN

            if span not in Z:
                Z[span] = FLOAT_MIN

            logging.debug('span: %d, %d' % span)
            for split in xrange(start + 1, end):
                left_span, right_span = ((start, split), (split, end))
                left_node, right_node = (sent[start:split], sent[split:end])

                left = node_map[left_span]
                right = node_map[right_span]

                node = model.node(left, right)
                prob_split = model.comp_unprob(node, left, right)
                Z_split = prob_split * Z[left_span] * Z[right_span] *  sentence_grammar[span][split]
                node_map[span] = Z_split.data * node
                Z[span] += Z_split
                comp_z += prob_split

            if Z[span].data > 0:
                node_map[span] /= Z[span].data

            Z[span] /= comp_z

    return -F.log2(Z[(0, length)]) + length * F.log2(model.z_leaf)

def evaluate(model, test_sents):
    m = model.copy()
    m.clear_z_leaf()
    m.init_z_leaf()
    m.volatile = True
    result = collections.defaultdict(lambda: 0)
    entropy = 0
    for sentence in test_sents:
        sentence = sentence.strip()
        hash = hashlib.md5(sentence).hexdigest()
        sentence_grammar = grammar[hash]
        indexed_sentence = indexer.index(sentence)
        entropy += traverse(m, indexed_sentence, len(indexed_sentence), sentence_grammar, train=False)

    return entropy.data

#vocab = {}
if args.test:
    max_size = 10
else:
    max_size = None


dataset_train='./data/ptb.train.txt_3500'
dataset_valid='./data/ptb.valid.txt_500'
dataset_test='./data/ptb.test.txt_1000'

# dataset_train='./data/train10'
# dataset_valid='./data/train10'
# dataset_test='./data/train10'

indexer = Indexer(dataset_train)
indexer.build_vocab()

vocab = indexer.get_vocab()
n_vocab = len(vocab)
Z_vocab = sum([grammar['vocab'][word] for word in vocab[1:]])
model = RecursiveNet(n_vocab, n_units)

if args.gpu >= 0:
    model.to_gpu()

# Setup optimizer
optimizer = optimizers.MomentumSGD(lr=learning_rate)
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.WeightDecay(l2_reg))

accum_loss = long()
batch_count = 0
start_at = time.time()
cur_at = start_at

metadata = grammar['metadata']
num_states = metadata['num_states']

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

for epoch in range(n_epoch):
    logging.info('Epoch: {0:d}'.format(epoch))
    epoch_count = 0
    total_loss = long()
    batch = 0
    cur_at = time.time()
    random.shuffle(train_sentences)

    model.init_z_leaf(True)
    for i, sentence in enumerate(train_sentences):
        sentence = sentence.strip()
        length = len(sentence.split())
        logging.debug('Processing sentence#{}:{}::{}'.format(i, length, sentence))

        hash = hashlib.md5(sentence).hexdigest()
        sentence_grammar = grammar[hash]
        indexed_sentence = indexer.index(sentence)
        loss = traverse(model, indexed_sentence, length, sentence_grammar, train=True)

        if not xp.isfinite(loss.data):
            logging.warn('Non finite loss for sentence: {}'.format(sentence))
            continue

        logging.debug('Processed sent#{}: {}'.format(epoch_count, loss.data))
        accum_loss += loss
        epoch_count += 1
        batch_count += 1

        if batch_count >= batchsize:
            logging.info('Updating batch gradient for batch: {}'.format(batch))
            batch += 1
            accum_loss /= batch_count
            model.cleargrads()
            try:
                accum_loss.backward()
            except Exception,e:
                logging.error("Failed update for batch %d\n%s" % (batch, e))
                accum_loss = 0

            total_loss += float(accum_loss.data)
            optimizer.update()

            accum_loss = 0
            batch_count = 0

            logging.info('loss: {:.5f}'.format(total_loss))
            model.clear_z_leaf()
            model.init_z_leaf(True)

    now = time.time()
    throughput = float(train_size) / (now - cur_at)
    logging.info('{:.2f} iters/sec, {:.2f} sec'.format(throughput, now - cur_at))

    if (epoch + 1) % epoch_per_eval == 0:
        logging.info('Train data evaluation: %.5f' % evaluate(model, train_sents))
        logging.info('Develop data evaluation: %.5f' % evaluate(model, valid_sents))
logging.info('Test evaluateion')
logging.info('Test loss: {:.5f}'.format(float(evaluate(model, test_sentences))))
