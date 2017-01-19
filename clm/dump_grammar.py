from edu.stanford.nlp.parser.lexparser import LexicalizedParser, Lexicon
from math import exp, log


import argparse
import shelve
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

                if state not in i_score[span]:
                    i_score[span][state] = {}

                rule = '%s %s' % (state, index)
                if rule not in i_score[span][state]:
                    i_score[span][state][rule] = 0

                i_score[span][state][rule] += exp(lexScore)

    def unaryScore(self, i_score, span):
        for state, rule_to_val in i_score[span].iteritems():
            for ur in self.ug.closedRulesByChild(state):
                pstate = ur.parent()
                ps = ur.score()

                if pstate not in i_score[span]:
                    i_score[span][pstate] = {}

                for rule, score  in rule_to_val.iteritems():
                    rule_split = rule.split()
                    rule_split[0] = str(pstate)
                    rule = " ".join(rule_split)

                    if rule not in i_score[span][pstate]:
                        i_score[span][pstate][rule] = 0

                    i_score[span][pstate][rule] += exp(log(score) + ps)

    def insideChartCell(self, i_score, start, diff):
        end = start + diff
        span = (start, end)

        for split in xrange(start + 1, end):
            left_span = (start, split)
            right_span = (split, end)

            for rstate in i_score[right_span].keys():
                for br in self.bg.splitRulesWithRC(rstate):
                    lstate = br.leftChild
                    pstate = br.parent()
                    rule_score = br.score()

                    if lstate not in i_score[left_span]:
                        continue

                    if span not in i_score:
                        i_score[span] = {}

                    if pstate not in i_score[span]:
                        i_score[span][pstate] = {}

                    rule = '%s %s %s' % (pstate, lstate, rstate)
                    if rule not in i_score[span][pstate]:
                        i_score[span][pstate][rule] = 0

                    i_score[span][pstate][rule] += exp(rule_score)

            for lstate in i_score[left_span].keys():
                for br in self.bg.splitRulesWithLC(lstate):
                    rstate = br.rightChild
                    pstate = br.parent()
                    rule_score = br.score()

                    if rstate not in i_score[right_span]:
                        continue

                    if span not in i_score:
                        i_score[span] = {}

                    if pstate not in i_score[span]:
                        i_score[span][pstate] = {}

                    rule = '%s %s %s' % (pstate, lstate, rstate)
                    if rule not in i_score[span][pstate]:
                        i_score[span][pstate][rule] = 0

                    i_score[span][pstate][rule] += exp(rule_score)

        # unary
        self.unaryScore(i_score, span)

    def calculateScore(self, input_filename, lower_case=False):

        if self.output_filename is None:
            self.output_filename = "%s.grammar" % input_filename

        Grmr = shelve.open(self.output_filename)
        with open(self.output_filename, 'w+') as fout:
            # write out metadata
            Grmr['metadata'] = {'version': 1.0,
                                'input_filename': input_filename,
                                'grammar': self.grammar,
                                'num_states': self.num_states
                                }

        with open(input_filename) as fin:
            for sentence in fin:
                sentence = sentence.strip()
                print "Processing Sentence: '%s'" % sentence

                print "Computing Lex Score"
                words = sentence.split()
                length = len(words)

                i_score = {}
                for word in [Word(i, word, self.getToken(word, i, lower_case)) \
                                for i, word in enumerate(words)]:
                    start = word.loc
                    end = start + 1
                    span = (start, end)

                    # Lex Score
                    self.lexScore(i_score, word, start, span)

                    # unary
                    self.unaryScore(i_score, span)

                print "Computing inside score"
                for diff in xrange(2, length + 1):
                    print "Compute inside score for span size: %d" % diff
                    for start in xrange(0, length - diff + 1):
                        self.insideChartCell(i_score, start, diff)

                print "Done computing inside score"

                Grmr[sentence] = i_score
                Grmr.sync()

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
