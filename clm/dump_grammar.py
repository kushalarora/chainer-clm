from edu.stanford.nlp.parser.lexparser import LexicalizedParser, Lexicon
from math import exp, log


import argparse
import shelve
import math
import os
import pickle
import hashlib

def getGrammar(grammar):
    dirname = os.path.dirname(os.path.realpath(__file__))
    return LexicalizedParser.loadModel(os.path.join(dirname, '../dependencies/%s.ser.gz' % grammar))

input_filename = 'data/train10'

class Word(object):
    def __init__(self, loc, word, signature):
        self.loc = loc
        self.word = word
        self.signature = signature

    def __str__(self):
        return "(%d %s %d)" %(self.loc, self.word, self.signature)

class InsideScorer(object):
    def __init__(self, grammar, output_filename=None):
        self.grammar = grammar
        lexicalizedParser = getGrammar(grammar)

        self.stateIndex = lexicalizedParser.stateIndex;
        self.wordIndex = lexicalizedParser.wordIndex;
        self.tagIndex = lexicalizedParser.tagIndex;

        self.goalStr = lexicalizedParser.treebankLanguagePack().startSymbol();
        self.bg = lexicalizedParser.bg;
        self.ug = lexicalizedParser.ug;
        self.lex = lexicalizedParser.lex;
        self.num_states = lexicalizedParser.stateIndex.size()
        self.output_filename = output_filename

    def getToken(self, word, loc, lower_case=False):
        index = -1
        signature = word

        if (lower_case):
            signature = word.lower()

        if (not self.wordIndex.contains(signature)):
            signature = word.lower()
            if (not self.wordIndex.contains(signature)):
                signature = word.title()
                if (not self.wordIndex.contains(signature)):
                    signature = self.lex.getUnknownWordModel().getSignature(word, loc)
        return signature

    def lexScore(self, i_score, span_split_score, vocab_scores, word, start, span):

        span_split_state_score = {}
        if span not in i_score:
            i_score[span] = {}
            span_split_score[span] = {}
            span_split_state_score[span] = {}

        split = span[0]
        if split not in span_split_score[span]:
            span_split_score[span][split] = 0
            span_split_state_score[span][split] = {}

        index = self.wordIndex.indexOf(word.signature)
        tagIter = self.lex.ruleIteratorByWord(index, start, None)
        while(tagIter.hasNext()):
            tag = tagIter.next()
            state = self.stateIndex.indexOf(self.tagIndex.get(tag.tag()))
            lexScore = self.lex.score(tag, start, word.signature, None)

            if (lexScore > float('-inf')):
                if state not in i_score[span]:
                    i_score[span][state] = 0

                if state not in span_split_state_score[span][split]:
                    span_split_state_score[span][split][state] = 0

                score = exp(lexScore)
                i_score[span][state] += score
                span_split_score[span][split] += score
                span_split_state_score[span][split][state] += score

        # unary
        self.unaryScore(span, split, i_score, span_split_score, span_split_state_score)

        if word.word not in vocab_scores:
            vocab_scores[word.word] = span_split_score[span][split]

    def unaryScore(self, span, split, i_score, span_split_score, span_split_state_score):
        state_list = span_split_state_score[span][split].keys()
        for state in state_list:
            for ur in self.ug.closedRulesByChild(state):
                pstate = ur.parent()
                ps = ur.score()

                if pstate not in i_score[span]:
                    i_score[span][pstate] = 0

                split_score = log(span_split_state_score[span][split][state])
                state_score = log(i_score[span][state])

                span_split_score[span][split] += exp(split_score + ps)
                i_score[span][pstate] += exp(state_score + ps)

    def insideChartCell(self, start, diff, i_score, span_split_score):
        end = start + diff
        span = (start, end)

        span_split_state_score = {}
        if span not in i_score:
            i_score[span] = {}
            span_split_score[span] = {}
            span_split_state_score[span] = {}

        for split in xrange(start + 1, end):
            if split not in span_split_score:
                span_split_score[span][split] = 0
                span_split_state_score[span][split] = {}

            left_span = (start, split)
            right_span = (split, end)

            state_list = i_score[right_span].keys()
            for rstate in state_list:
                for br in self.bg.splitRulesWithRC(rstate):
                    lstate = br.leftChild
                    pstate = br.parent()
                    rule_score = br.score()

                    if lstate not in i_score[left_span]:
                        continue

                    if pstate not in i_score[span]:
                        i_score[span][pstate] = 0

                    if pstate not in span_split_state_score[span][split]:
                        span_split_state_score[span][split][pstate] = 0

                    score = exp(rule_score +
                                log(i_score[left_span][lstate]) +
                                log(i_score[right_span][rstate]))
                    span_split_score[span][split] += score
                    span_split_state_score[span][split][pstate] += score
                    i_score[span][pstate] += score

            state_list = i_score[left_span].keys()
            for lstate in state_list:
                for br in self.bg.splitRulesWithLC(lstate):
                    rstate = br.rightChild
                    pstate = br.parent()
                    rule_score = br.score()

                    if rstate not in i_score[right_span]:
                        continue

                    if pstate not in i_score[span]:
                        i_score[span][pstate] = 0

                    if pstate not in span_split_state_score[span][split]:
                        span_split_state_score[span][split][pstate] = 0

                    score = exp(rule_score +
                                log(i_score[left_span][lstate]) +
                                log(i_score[right_span][rstate]))
                    span_split_score[span][split] += score
                    span_split_state_score[span][split][pstate] += score
                    i_score[span][pstate] += score

            # unary
            self.unaryScore(span, split, i_score, span_split_score, span_split_state_score)

        z_split = sum(span_split_score[span].values())
        if z_split == 0:
            return

        for split in span_split_score[span]:
            span_split_score[span][split] /= z_split

    def calculateScore(self, input_filename, lower_case=False):

        if self.output_filename is None:
            self.output_filename = "%s.grammar" % input_filename

        Grmr = shelve.open(self.output_filename)

        # write out metadata
        Grmr['metadata'] = {'version': 1.0,
                            'input_filename': input_filename,
                            'grammar': self.grammar,
                            'num_states': self.num_states
                            }
        Grmr['scores'] = {}
        Grmr['vocab'] = {}
        sent_count = 0
        with open(input_filename) as fin:
            score_dict = Grmr['scores']
            vocab_scores = Grmr['vocab']
            for sentence in fin:
                sentence = sentence.strip()
                print "Processing Sentence(%d): '%s'" % (sent_count, sentence)
                sent_count += 1

                print "Computing Lex Score"
                words = sentence.split()
                length = len(words)

                span_split_score = {}
                i_score = {}
                for word in [Word(i, word, self.getToken(word, i, lower_case)) \
                                for i, word in enumerate(words)]:
                    start = word.loc
                    end = start + 1
                    span = (start, end)

                    # Lex Score
                    self.lexScore(i_score, span_split_score, vocab_scores, word, start, span)

                print "Computing inside score"
                for diff in xrange(2, length + 1):
                    print "Compute inside score for span size: %d" % diff
                    for start in xrange(0, length - diff + 1):
                        self.insideChartCell(start, diff, i_score, span_split_score)

                print "Done computing inside score"

                hash = hashlib.md5(sentence).hexdigest()
                Grmr[hash] = span_split_score

                Grmr.sync()
            Grmr['vocab'] = vocab_scores
        Grmr.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_filename', '-f', type=str, help='input file for calculating iscores')
    parser.add_argument('--grammar', '-g', type=str, default='englishPCFG', help='Stanford Grammar file')
    parser.add_argument('--output_filename', '-o', type=str, help='Pickled i_scores')
    parser.add_argument('--lower_case', help='lowercase input', action='store_true')
    parser.set_defaults(lower_case=False)

    args = parser.parse_args()

    iscorer = InsideScorer(args.grammar, args.output_filename)
    i_scores = iscorer.calculateScore(args.input_filename, args.lower_case)
