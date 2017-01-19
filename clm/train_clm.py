#!/usr/bin/env python
"""
"""

import argparse
import codecs
import collections
import os
import random
import re
import shelve
import time

import numpy as np

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
from datasets import Indexer


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--epoch', '-e', default=400, type=int,
                    help='number of epochs to learn')
parser.add_argument('--unit', '-u', default=10, type=int,
                    help='number of units')
parser.add_argument('--batchsize', '-b', type=int, default=20,
                    help='learning minibatch size')
parser.add_argument('--label', '-l', type=int, default=5,
                    help='number of labels')
parser.add_argument('--epocheval', '-p', type=int, default=5,
                    help='number of epochs per evaluation')
parser.add_argument('--grammar_filename', '-gp', type=str,
                    help='grammar object by running dump_grammar.py')

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

# Get preprocessed Grammar
grammar = None
if os.path.exists(args.grammar_filename):
    grammar = shelve.open(args.grammar_filename)

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

def traverse(model, sent, length, sentence_grammar, train=True):
    assert isinstance(sent, list)

    Z = {}
    node_map = {}

    # leaf nodes
    for index in xrange(0, length):
        span = (index, index + 1)
        print('Processing index: (%d-%d)' % span)
        word = xp.array([sent[index]], np.int32)
        loss = 0
        x = chainer.Variable(word, volatile=not train)
        node = model.leaf(x)

        node_map[span] = node

        if span not in Z:
            Z[span] = {}

        inside = model.leaf_unprob(node)/model.z_leaf
        for state, rule_dict in sentence_grammar[span].iteritems():
            for rule, theta in rule_dict.iteritems():
                if state not in Z[span]:
                    Z[span][state] = 0
                Z[span][state] += inside * theta

    # internal nodes
    for diff in xrange(2, length + 1):
        for start in xrange(0, length - diff + 1):
            end = start + diff

            span = (start, end)
            print('Processing index: (%d-%d)' % span)

            comp_z = 0
            pstates = set([])

            for split in xrange(start + 1, end):
                left_span, right_span = ((start, split), (split, end))
                left_node, right_node = (sent[start:split], sent[split:end])

                left = node_map[left_span]
                right = node_map[right_span]

                node = model.node(left, right)

                node_map[span] = node

                prob_split = model.comp_unprob(node, left, right)

                comp_z += prob_split

                if span not in Z:
                    Z[span] = {}

                for state, rule_dict in sentence_grammar[span].iteritems():
                    for rule, theta in rule_dict.iteritems():
                        pstate, lstate, rstate = map(lambda a: int(a), rule.split())

                        if lstate not in Z[left_span]:
                            continue

                        if rstate not in Z[right_span]:
                            continue

                        if pstate not in Z[span]:
                            Z[span][pstate] = 0

                        Z[span][pstate] += prob_split * theta * \
                                                    Z[left_span][lstate] * \
                                                    Z[right_span][rstate]
                        pstates.add(pstate)

            for pstate in pstates:
                Z[span][pstate] = Z[span][pstate]/comp_z


    prob_sent = 0
    for prob in Z[(0, length)].values():
        prob_sent += prob

    return prob_sent

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


dataset_train='./data/train10'
dataset_valid='./data/valid10'
dataset_test='./data/valid10'

indexer = Indexer(dataset_train)
indexer.build_vocab()

vocab = indexer.get_vocab()
n_vocab = len(vocab)

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

metadata = grammar['metadata']
num_states = metadata['num_states']

print('Vocab size: {}'.format(n_vocab))

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

print('Training dataset size: {}'.format(train_size))
print('Valid dataset size: {}'.format(valid_size))
print('Test dataset size: {}'.format(test_size))

for epoch in range(n_epoch):
    print('Epoch: {0:d}'.format(epoch))
    epoch_count = 0
    total_loss = 0
    batch = 0
    cur_at = time.time()
    random.shuffle(train_sentences)

    model.init_z_leaf(True)

    for i, sentence in enumerate(train_sentences):
        sentence = sentence.strip()
        length = len(sentence.split())
        print('Processing sentence#{}. len={}'.format(i, length))
        sentence_grammar = grammar[sentence]

        Z = {}
        indexed_sentence = indexer.index(sentence)

        prob_sent = traverse(model, indexed_sentence, length, sentence_grammar, train=True)

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
    throughput = float(train_size) / (now - cur_at)
    print('{:.2f} iters/sec, {:.2f} sec'.format(throughput, now - cur_at))
    print()

    if False and (epoch + 1) % epoch_per_eval == 0:
        print('Train data evaluation:')
        print(evaluate(model, train_sents))
        print('Develop data evaluation:')
        print(evaluate(model, valid_sents))
        print('')
print('Test evaluateion')
#print(evaluate(model, test_sents))
