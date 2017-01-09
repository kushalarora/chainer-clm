from edu.stanford.nlp.parser.lexparser import LexicalizedParser
from math import exp, log


import argparse
import math
import os
import pickle

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
    def __init__(self, grammar):
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

    def lexScore(self, i_score, word, start, span):
        index = self.wordIndex.indexOf(word.signature)
        tagIter = self.lex.ruleIteratorByWord(index, start, None)
        while(tagIter.hasNext()):
            tag = tagIter.next()
            state = self.stateIndex.indexOf(self.tagIndex.get(tag.tag()))
            lexScore = self.lex.score(tag, start, word.signature, None)

            if (lexScore > float('-inf')):
                if span not in i_score:
                    i_score[span] = {}
                i_score[span][state] = exp(lexScore)

    def unaryScore(self, i_score, span):
        for state in i_score[span].keys():
            is_val = log(i_score[span][state])

            for ur in self.ug.closedRulesByChild(state):
                parentState = ur.parent()
                ps = ur.score()
                i_score[span][parentState] = exp(is_val + ps)


    def insideChartCell(self, i_score, start, diff):
        end = start + diff
        span = (start, end)

        for split in xrange(start + 1, end):
            right_span = (start, split)
            left_span = (split, end)

            for rstate in i_score[right_span].keys():
                for br in self.bg.splitRulesWithRC(rstate):
                    lstate = br.leftChild
                    pstate = br.parent()
                    rule_score = br.score()

                    if lstate not in i_score[left_span].keys():
                        continue

                    if span not in i_score:
                        i_score[span] = {}
                    i_score[span][pstate] = exp(rule_score)

            for lstate in i_score[left_span].keys():
                for br in self.bg.splitRulesWithLC(lstate):
                    rstate = br.rightChild
                    pstate = br.parent()
                    rule_score = br.score()

                    if rstate not in i_score[right_span].keys():
                        continue

                    if span not in i_score:
                        i_score[span] = {}
                    i_score[span][pstate] = exp(rule_score)

        # unary
        self.unaryScore(i_score, span)

    def calculateScore(self, input_filename):
        i_scores = {}
        with open(input_filename) as f:
            for sentence in f:
                sentence = sentence.strip()
                print "Processing Sentence: '%s'(%d)" % (sentence.lower(), len(sentence.split()))
                length = len(sentence.split())
                i_score = {}
                print "Computing Lex Score"
                for word in [Word(i, word, self.getToken(word, i)) \
                                for i, word in enumerate(sentence.split())]:
                    start = word.loc
                    end = start + 1
                    span = (start, end)

                    # Lex Score
                    self.lexScore(i_score, word, start, span)

                    # unary
                    self.unaryScore(i_score, span)

                print "Computing inside score"
                for diff in xrange(2, length):
                    print "Compute inside score for span size: %d" % diff
                    for start in xrange(0, length - diff):
                        self.insideChartCell(i_score, start, diff)

                print "Done computing inside score"
                i_scores[sentence] = i_score
            return i_scores

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_filename', '-f', type=str, help='input file for calculating iscores')
    parser.add_argument('--grammar', '-g', type=str, default='englishPCFG', help='Stanford Grammar file')
    parser.add_argument('--output_filename', '-o', type=str, help='Pickled i_scores', default='data/grammar.p')

    args = parser.parse_args()
    iscorer = InsideScorer(args.grammar)
    i_scores = iscorer.calculateScore(args.input_filename)

    pickle.dump(i_scores, open(args.output_filename, 'wb'))
