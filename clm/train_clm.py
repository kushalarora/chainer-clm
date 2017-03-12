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
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
from datasets import Indexer

from sklearn.decomposition import PCA
from gensim.models import KeyedVectors


FLOAT_MIN = 1e-30
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
grammar = shelve.open(args.grammar_filename)


class RecursiveNet(chainer.Chain):
    def __init__(self, n_vocab, n_units, embeddings):
        super(RecursiveNet, self).__init__(
            embed=L.EmbedID(n_vocab, n_units, initialW=embeddings),
            # E=L.Linear(300, n_units),
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

    def leaf_energy(self, v):
        return self.u(v)

    def leaf_unprob(self, v):
        return F.exp(-1 * self.leaf_energy(v))

    def comp_energy(self, parent, left, right):
        return self.u(parent) + self.h1(left) + self.h2(right)

    def comp_unprob(self, parent, left, right):
        return F.exp(-1 * self.comp_energy(parent, left, right))

    def init_z_leaf(self, train=False):
            words = xp.array(xrange(self.n_vocab), np.int32)
            X = chainer.Variable(words, volatile=not train)
            self.z_leaf = F.broadcast_to(F.sum(self.leaf_unprob(self.leaf(X))), (1,1))


    def clear_z_leaf(self):
        self.z_leaf = FLOAT_MIN


def calcZ(model, sent, length, train=False):
    assert isinstance(sent, list)

    if not train:
        model = model.copy()

    Z = {}
    X = {}
    # leaf nodes
    for index in xrange(0, length):
        span = (index, index + 1)
        split = index

        if span not in X:
            X[span] = {0: 0}

        if span not in Z:
            Z[span] = {0: 0}

        word = xp.array([sent[index]], np.int32)
        x = chainer.Variable(word, volatile=not train)
        node = model.leaf(x)

        X[span][0] = node
        Z[span][0] = model.leaf_unprob(node)

    # internal nodes
    for diff in xrange(2, length + 1):
        for start in xrange(length - diff + 1):
            end = start + diff
            span = (start, end)

            if span not in X:
                X[span] = {0: 0}

            if span not in Z:
                Z[span] = {0: 0}

            logging.debug('span: %d, %d' % span)
            for split in xrange(start + 1, end):
                left_span, right_span = ((start, split), (split, end))
                left, right = X[left_span][0], X[right_span][0]

                X[span][split] = model.node(left, right)
                Z[span][split] = model.comp_unprob(node, left, right)

                X[span][0] = Z[span][split].data * node
                Z[span][0] += Z[span][split]

            if Z[span][0].data > 0:
                X[span][0] /= Z[span][0].data

            for split in xrange(start + 1, end):
                if Z[span][0].data > 0:
                    Z[span][split] /= Z[span][0]

    return Z, X


def inside(model, sent, length, sentence_grammar, Z, train=False):
    A = {}
    # leaf nodes
    for index in xrange(0, length):
        span = (index, index + 1)
        split = index

        if span not in A:
            A[span] = {0: 0}

        A[span][0] = Z[span][0].data

    # internal nodes
    for diff in xrange(2, length + 1):
        for start in xrange(length - diff + 1):
            end = start + diff
            span = (start, end)

            if span not in A:
                A[span] = {0: 0}

            logging.debug('span: %d, %d' % span)
            for split in xrange(start + 1, end):
                left_span, right_span = ((start, split), (split, end))
                A[span][split] = Z[span][split].data * \
                    sentence_grammar[span][split] * \
                    A[left_span][0] * \
                    A[right_span][0]

                A[span][0] += A[span][split]
    return A


def outside(model, A, Z, X, length):
    B = {(0, length): xp.array([[1.0]])}
    for diff in reversed(xrange(1, length + 1)):
        for start in xrange(length + 1 - diff):
            end = start + diff
            span = (start, end)
            for split in xrange(start + 1, end):
                left_span, right_span = ((start, split), (split, end))
                B[left_span] = B[span] * Z[span][split].data * A[right_span][0]
                B[right_span] = B[span] * Z[span][split].data * A[left_span][0]
    return B


def mu(A, B, length):
    Mu = {}
    for index in xrange(0, length):
        span = (index, index + 1)
        split = index

        if span not in Mu:
            Mu[span] = {}

        Mu[span][0] = A[span][0] * B[span][0]

    for diff in xrange(2, length + 1):
        for start in xrange(length + 1 - diff):
            end = start + diff
            span = (start, end)

            if span not in Mu:
                Mu[span] = {}

            Mu[span][0] = A[span][0] * B[span][0]
            for split in xrange(start + 1, end):
                Mu[span][split] = A[span][split] * B[span][0]
    return Mu


def EStep(model, sent, length, sentence_grammar):
    Z, X = calcZ(model, sent, length, train=False)
    A = inside(model, sent, length, sentence_grammar, Z, train=True)
    B = outside(model, A, Z, X, length)
    Mu = mu(A, B, length)
    return A, Mu


def MStep(model, Z, Mu, A, length):
        loss = 0
        for index in xrange(0, length):
            span = (index, index + 1)
            split = index

            loss += (-F.log(Z[span][0]) + F.log(model.z_leaf)) * Mu[span][0]

        for diff in xrange(2, length + 1):
            for start in xrange(0, length + 1 - diff):
                end = start + diff
                span = (start, end)

                for split in xrange(start + 1, end):
                    loss += -F.log(Z[span][split]) * Mu[span][split]

        return loss/A[(0, length)][0]


def evaluate(model, test_sents):
    m = model.copy()
    m.clear_z_leaf()
    m.init_z_leaf()
    m.volatile = True
    entropy = 0
    for sentence in test_sents:
        sentence = sentence.strip()
        hash = hashlib.md5(sentence).hexdigest()
        sentence_grammar = grammar[hash]
        indexed_sentence = indexer.index(sentence)
        length = len(indexed_sentence)
        Z, X = calcZ(model, indexed_sentence, length, train=False)
        A = inside(m, indexed_sentence,
                   length, sentence_grammar,
                   Z, train=False)

        entropy += -np.log2(A[0, length][0]) + \
            length * np.log2(m.z_leaf.data)

    return entropy


def embedding_vectors(n_units, vocab, word2vec_file):
    model = KeyedVectors.load_word2vec_format(word2vec_file, binary=True)
    model_vocab = ['unk' if word not in model else word for word in vocab]
    emb_300 = model[model_vocab]
    # return emb_300
    pca = PCA(n_components=n_units)
    return pca.fit_transform(emb_300)

# vocab = {}


if args.test:
    max_size = 10
else:
    max_size = None


dataset_train = './data/ptb.valid.txt_500'
dataset_valid = './data/ptb.valid.txt_500'
dataset_test = './data/ptb.test.txt_1000'
word2vec_file = '~/GoogleNews-vectors-negative300.bin'

indexer = Indexer(dataset_train)
indexer.build_vocab()

vocab = indexer.get_vocab()
n_vocab = len(vocab)
Z_vocab = sum([grammar['vocab'][word] for word in vocab[1:]])
# initW = embedding_vectors(n_units, vocab, word2vec_file)
initW = xp.random.uniform(-1, 1, (n_vocab, n_units))
model = RecursiveNet(n_vocab, n_units, initW)

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

#logging.info('Train data evaluation: %.5f'
#             % evaluate(model, train_sentences))

for epoch in range(n_epoch):
    logging.info('Epoch: {0:d}'.format(epoch))
    epoch_count = 0
    total_loss = long()
    batch = 0
    cur_at = time.time()
    random.shuffle(train_sentences)

    model.init_z_leaf(True)
    As = {}
    Mus = {}
    sentences = {}
    for i, sentence in enumerate(train_sentences):
        sentence = sentence.strip()
        length = len(sentence.split())
        logging.info('Processing sentence#{}:{}::{}'
                     .format(i, length, sentence))

        hash = hashlib.md5(sentence).hexdigest()
        sentence_grammar = grammar[hash]
        indexed_sentence = indexer.index(sentence)
        A, Mu = EStep(model, indexed_sentence, length, sentence_grammar)

        As[hash] = A
        Mus[hash] = Mu
        sentences[hash] = indexed_sentence
        epoch_count += 1
        batch_count += 1

        if batch_count >= batchsize:
            for em_epoch in xrange(n_em_epoch):
                logging.info('EM epoch: {0:d}'.format(em_epoch))
                for hash, sentence in sentences.iteritems():
                    length = len(sentence)
                    Z, X = calcZ(model, sentence, length, train=True)
                    A = As[hash]
                    Mu = Mus[hash]

                    loss = MStep(model, Z, Mu, A, length)
                    logging.debug('Processed sentence#{}:{}=>{}'
                                  .format(epoch_count, em_epoch, loss.data))
                    accum_loss += loss

                logging.info('Updating batch gradient for batch: {}'
                             .format(batch))
                batch += 1
                logging.info('loss: {:.5f}'.format(float(accum_loss.data)))
                model.cleargrads()
                try:
                    accum_loss.backward()
                except Exception, e:
                    logging.error("Failed update for batch %d\n%s"
                                  % (batch, e))
                    accum_loss = 0

                optimizer.update()

                accum_loss = 0

                model.clear_z_leaf()
                model.init_z_leaf(True)
            batch_count = 0
            As.clear()
            Mus.clear()
            sentences.clear()

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
