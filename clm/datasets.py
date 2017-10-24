import numpy as np

class Indexer(object):
    def __init__(self, input_filename):
        self.input_filename = input_filename

    def build_vocab(self):
        self.vocab = [None]
        self.vocab_dict = {None: 0}
        fin = open(self.input_filename)
        if fin is None:
            raise IOError("Filename %s not found" % filename)

        index = 1
        for line in fin:
            words = self.tokenize(line) 
            for i, word in enumerate(words):
                if word not in self.vocab_dict:
                    self.vocab_dict[word] = index
                    self.vocab.append(word)
                    index += 1

    def get_vocab(self):
        return self.vocab

    def index(self, sentence):
        if self.vocab_dict is None:
            raise BaseException("No vocabulary found. Call build_vocab method first.")

        indexes = []
        words = self.tokenize(sentence) 
        for word in words:
            try:
                id = self.vocab_dict[word]
            except KeyError:
                word = None
                id = 0;

            if id is None:
                raise RuntimeError(
                    "Word: %s missing in vocabulary" % word)

            indexes.append(id)
        return indexes

    def tokenize(self, sentence):
        return sentence.strip().split() + ['<eos>'] 

