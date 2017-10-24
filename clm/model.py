import chainer
import chainer.computational_graph as c
import chainer.functions as F
import chainer.links as L
import logging
import numpy as np

FLOAT_MIN = 1e-30

class RecursiveNet(chainer.Chain):
    def __init__(self, n_vocab, n_units, embeddings):
        super(RecursiveNet, self).__init__(
            embed=L.EmbedID(n_vocab, n_units, initialW=embeddings),
            w1=L.Linear(n_units, n_units),
            w2=L.Linear(n_units, n_units),
            u=L.Linear(n_units, 1),
            h1=L.Linear(n_units, 1),
            h2=L.Linear(n_units, 1))
        self.n_vocab = n_vocab
        self.n_units = n_units
        self.z_leaf = FLOAT_MIN

    def leaf(self, x):
        return self.embed(x)

    def node(self, left, right):
        return F.tanh(self.w1(left) + self.w2(right))

    def leaf_energy(self, v):
        return self.u(v)

    def leaf_unprob(self, v):
        return F.exp(-1 * self.leaf_energy(v))

    def comp_energy(self, parent, left, right):
        return self.u(parent) + self.h1(left) + self.h2(right)

    def comp_unprob(self, parent, left, right):
        return F.exp(-1 * self.comp_energy(parent, left, right))

    def init_z_leaf(self, words):
        self.z_leaf = F.sum( \
                        self.leaf_unprob( \
                            self.leaf(chainer.Variable(words))), 
                        keepdims=True)

    def clear_z_leaf(self):
        self.z_leaf = FLOAT_MIN

    def __call__(self, input):
        sent, sentence_grammar = input
        length = len(sent)

        # leaf nodes
        node_batch = self.leaf(chainer.Variable(sent))
        Z_batch = self.leaf_unprob(node_batch)

        A = {};X = {};
        for index in xrange(0, length):
            span = (index, index + 1)
            split = index

            if span not in X:
                X[span] = 0

            if span not in A:
                A[span] = 0 

            # Indexing this way as this keeps the shape
            X[span] = node_batch[index:index+1]
            A[span] = Z_batch[index:index+1]/self.z_leaf

        # internal nodes
        for diff in xrange(2, length + 1):

            # Batch Z and X for (start, end)
            lefts = [];rights = [];positions = [];
            for start in xrange(length - diff + 1):
                end = start + diff
                span = (start, end)

                logging.debug('span: %d, %d' % span)
                for split in xrange(start + 1, end):
                    left_span, right_span = ((start, split), (split, end))
                    lefts.append(X[left_span])
                    rights.append(X[right_span])
                    positions.append(((start, end), split))

            if len(lefts) > 1:
                left_batch = F.concat(tuple(lefts), axis=0)
            else:
                left_batch = lefts[0]
            if len(rights) > 1:
                right_batch = F.concat(tuple(rights), axis=0)
            else:
                right_batch = rights[0]

            nodes_batch = self.node(left_batch, right_batch)
            Z_batch = self.comp_unprob(nodes_batch, left_batch, right_batch)
            
            Z_sum = {}
            for i, span_split in enumerate(positions):
                span, split = span_split
                start, end = span

                if span not in X:
                    X[span] = 0

                if span not in A:
                    A[span] = 0 
                
                if span not in Z_sum:
                    Z_sum[span] = 0
                
                Z = Z_batch[i:i+1]
                X[span] += Z.data * nodes_batch[i:i+1]
                Z_sum[span] += Z

                left_span, right_span = ((start, split), (split, end))
                A[span] += Z * A[left_span] * A[right_span] * \
                            sentence_grammar.get(span, {}).get(split, 0)

            for start in xrange(length - diff + 1):
                end = start + diff
                span = (start, end)

                if Z_sum[span].data == 0:
                    continue

                X[span] /= Z_sum[span].data

                A[span] /= Z_sum[span]

        return A[(0, length)] 

class MaxMarginTrainer(chainer.Chain):
    def __init__(self, recursive_net):
        self.net=recursive_net

    def __call__(self, input, distorted):
        sent_inp, grammar_inp = input
        sent_dist, grammar_dist = distorted
        return F.relu(self.net(distorted) - self.net(input) - 1)

if __name__ == "__main__":
    
    initW = np.random.uniform(-1, 1, (100, 10))
    model = RecursiveNet(100, 10, initW) 
    
    words = np.array(xrange(100), np.int32)

    trainer = MaxMarginTrainer(model)
    
    for i in xrange(10):
        model.init_z_leaf(words)
        loss = trainer((np.array([1,2,3], np.int32),
                        {(0,1): {0 : 1}, 
                         (0,2): {1 : 1},
                         (0,3): {2 : 1}, 
                         (0,4): {3 : 1}}), 
                        (np.array([1,4,3], np.int32), 
                            {(0,1): {0 : 1}, 
                            (0,2): {1 : 1},
                            (0,3): {2 : 1}, 
                            (0,4): {3 : 1}}))
        model.cleargrads()

        loss.backward()

        model.clear_z_leaf()

