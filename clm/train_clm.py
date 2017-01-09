#!/usr/bin/env python
"""
"""

import argparse
import codecs
import collections
import random
import re
import time

import numpy as np

import chainer
from chainer import cuda
import chainer.functions as F
from chainer.functions.math import sum
import chainer.links as L
from chainer import optimizers
from datasets import build_vocab, index_data


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--epoch', '-e', default=400, type=int,
                    help='number of epochs to learn')
parser.add_argument('--unit', '-u', default=30, type=int,
                    help='number of units')
parser.add_argument('--batchsize', '-b', type=int, default=10,
                    help='learning minibatch size')
parser.add_argument('--label', '-l', type=int, default=5,
                    help='number of labels')
parser.add_argument('--epocheval', '-p', type=int, default=5,
                    help='number of epochs per evaluation')
parser.add_argument('--test', dest='test', action='store_true')
parser.set_defaults(test=False)
args = parser.parse_args()
if args.gpu >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if args.gpu >= 0 else np

n_epoch = args.epoch       # number of epochs
n_units = args.unit        # number of units per layer
batchsize = args.batchsize      # minibatch size
n_label = args.label         # number of labels
epoch_per_eval = args.epocheval  # number of epochs per evaluation


class SexpParser(object):

    def __init__(self, line):
        self.tokens = re.findall(r'\(|\)|[^\(\) ]+', line)
        self.pos = 0

    def parse(self):
        assert self.pos < len(self.tokens)
        token = self.tokens[self.pos]
        assert token != ')'
        self.pos += 1

        if token == '(':
            children = []
            while True:
                assert self.pos < len(self.tokens)
                if self.tokens[self.pos] == ')':
                    self.pos += 1
                    break
                else:
                    children.append(self.parse())
            return children
        else:
            return token


def convert_tree(vocab, exp):
    assert isinstance(exp, list) and (len(exp) == 2 or len(exp) == 3)

    if len(exp) == 2:
        label, leaf = exp
        if leaf not in vocab:
            vocab[leaf] = len(vocab)
        return {'label': int(label), 'node': vocab[leaf]}
    elif len(exp) == 3:
        label, left, right = exp
        node = (convert_tree(vocab, left), convert_tree(vocab, right))
        return {'label': int(label), 'node': node}


def read_corpus(path, vocab, max_size):
    with codecs.open(path, encoding='utf-8') as f:
        trees = []
        for line in f:
            line = line.strip()
            tree = SexpParser(line).parse()
            trees.append(convert_tree(vocab, tree))
            if max_size and len(trees) >= max_size:
                break

        return trees


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
        self.z_leaf = 0

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
        self.z_leaf = 0

def traverse(model, sent, start, end, train=True):
    assert isinstance(sent, list)
    if len(sent[start:end]) == 1:
        # leaf node
        word = xp.array(sent[start:end], np.int32)
        loss = 0
        x = chainer.Variable(word, volatile=not train)
        node = model.leaf(x)
        inside = model.leaf_unprob(node)/model.z_leaf
    else:
        # internal node
        comp_z = 0
        inside = 0
        for split in xrange(start + 1, end):
            left_node, right_node = (sent[start:split], sent[split:end])
            left_inside, left = traverse(
                model, sent, start, split, train=train)

            right_inside, right = traverse(
                model, sent, split, end, train=train)

            node = model.node(left, right)

            prob_split = model.comp_unprob(node, left, right)

            comp_z += prob_split

            inside += right_inside * left_inside * prob_split

        inside = inside/comp_z

    return inside, node

def evaluate(model, test_sents):
    m = model.copy()
    m.clear_z_leaf()
    m.init_z_leaf()
    m.volatile = True
    result = collections.defaultdict(lambda: 0)
    entropy = 0
    for sent in test_sents:
        inside, node = traverse(m, sent, 0, len(sent), train=False)
        entropy += -F.log2(inside).data

    return entropy/len(test_sents)

#vocab = {}
if args.test:
    max_size = 10
else:
    max_size = None


dataset_train='../data/train10'
dataset_valid='../data/valid10'
dataset_test='../data/valid10'

train_sents, vocab, vocab_index = build_vocab(dataset_train)
valid_sents = index_data(dataset_valid, vocab_index)
test_sents = index_data(dataset_test, vocab_index)

n_vocab = len(vocab)

print('Vocab size: {}'.format(len(vocab)))
print('Training dataset size: {}'.format(len(train_sents)))
print('Valid dataset size: {}'.format(len(valid_sents)))
print('Test dataset size: {}'.format(len(test_sents)))

model = RecursiveNet(n_vocab, n_units)


if args.gpu >= 0:
    model.to_gpu()

# Setup optimizer
optimizer = optimizers.AdaGrad(lr=0.1)
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.WeightDecay(0.001))

accum_loss = 0
batch_count = 0
start_at = time.time()
cur_at = start_at
for epoch in range(n_epoch):
    print('Epoch: {0:d}'.format(epoch))
    epoch_count = 0
    total_loss = 0
    batch = 0
    cur_at = time.time()
    random.shuffle(train_sents)

    model.init_z_leaf(True)

    for sent in train_sents:
        print('Processing sent#{}. len={}'.format(epoch_count, len(sent)))
        prob_sent, v = traverse(model, sent, 0, len(sent), train=True)
        print('Processed sent#{}'.format(epoch_count))
        loss = -1 * F.log2(prob_sent)
        accum_loss += loss
        epoch_count += 1
        batch_count += 1

        if batch_count >= batchsize:
            print('Updating batch gradient for batch: {}'.format(batch))
            batch += 1
            total_loss += float(accum_loss.data)
            accum_loss /= batch_count
            model.cleargrads()
            accum_loss.backward()
            optimizer.update()

            accum_loss = 0
            batch_count = 0

            print('loss: {:.2f}'.format(total_loss/epoch_count))
            model.clear_z_leaf()
            model.init_z_leaf(True)

    now = time.time()
    throughput = float(len(train_sents)) / (now - cur_at)
    print('{:.2f} iters/sec, {:.2f} sec'.format(throughput, now - cur_at))
    print()

    if (epoch + 1) % epoch_per_eval == 0:
        print('Train data evaluation:')
        print(evaluate(model, train_sents))
        print('Develop data evaluation:')
        print(evaluate(model, valid_sents))
        print('')

print('Test evaluateion')
print(evaluate(model, test_sents))
