#!/usr/bin/python3.5
# coding: utf-8
'''
1) calculate % of non-projective arcs (num of arcs = num of words in a sent) and
2) % of sents (= num of trees) with non-proj arcs
3) python ~/PycharmProjects/udpipe_conllu/tools-master/nonprojectivity.py --stats ~/udpipe/ud-treebanks-v2.0/UD_Russian-SynTagRus/ru_syntagrus-ud-train.conllu
'''

#from __future__ import division
import sys
import file_util # in_out fuction called from this script parses the script argvs and load the data, trees function parses the comments in the input file
# the list of conllu fieldnames: "ID,FORM,LEMMA,CPOSTAG,POSTAG,FEATS,HEAD,DEPREL,DEPS,MISC"
from file_util import ID,HEAD #column index for the columns we'll need
import argparse
import os
import itertools
import traceback
import random

# pass a conllu file in the working dir as input
THISDIR=os.path.dirname(os.path.abspath(__file__))

class Stats(object):
    def __init__(self):
        self.word_count = 0 # number of lines=words in a tree is the number of rels=arcs
        self.tree_count = 0 # number of sentences


    def count_cols(self,cols):
        if cols[ID].isdigit() or u"." in cols[ID]: #word or empty word
            self.word_count+=1

    def print_basic_stats(self,out):
        print("Tree count: ", self.tree_count)
        print("Word (=arcs) count: ", self.word_count)

if __name__=="__main__":
    opt_parser = argparse.ArgumentParser(description='Script for basic stats generation. Assumes a validated input.')
    opt_parser.add_argument('input', nargs='+', help='Input file name (can be several files), or "-" or nothing for standard input.')
    opt_parser.add_argument('--stats',action='store_true',default=False, help='Print basic stats')
    args = opt_parser.parse_args()  # Parsed command-line arguments
    args.output = "-"
    inp, out = file_util.in_out(args, multiple_files=True)
    trees = file_util.trees(inp)

    stats = Stats()
    # counters for the whole treebank
    count = 0 # counter for non-projective dependencies, i.e. arcs that start or finish within other arcs, but not both (the latter would be nested arcs which is fine)
    nonpr_sents = 0  # counter for sents with any num of non-projective deps
    errors = 0  # counter for alleged annotation errors
    try:
        for comments, tree in trees: # i.e. for a sent in the conllu file
            stats.tree_count += 1
            #counters for the current sentence
            in_sent = 0 # resettable counter to mark that the current sent has a non-projective dependency
            tree_wc = 0 #counter for words in the current tree

            tree_arc_count = []  # list of all ID,HEAD pairs in a tree; it needs to be reset before processing every next tree
            for cols in tree: #tree is a list of token/word lines of the current tree
                #print cols
                stats.count_cols(cols) #this counts all words in all sentences (it is the part of code borrowed from tools-master, which makes little sense already)
                tree_wc +=1
                if '.' in cols[ID] and '_' in cols[HEAD]: #it is <type 'unicode'>
                    continue
                else:

                    try:
                        a = int(round(float(cols[ID]),0))
                        b = int(round(float(cols[HEAD]),0))
                        if a > b:
                            tree_arc_count.append([b,a])
                            #print( 'Reversed', '\t', [b,a] )
                        else:
                            tree_arc_count.append([a,b])  # this is a list of all arcs defined as 2-member-lists of start-end elements put in the right order: [[1, 2], [2, 18]]
                        #print( 'Born straight', '\t', [a,b] )
                    except ValueError:
                        print(cols[ID], cols[HEAD])
            # print >> out,'Lets inspect a random sample from the list of arcs: ', tree_arc_count  # random.sample(self.tree_arc_count, 2))  # self.tree_arc_count[4: 10]

            pairs = []  # produce a list of all pairwise combinations of arcs of type AB AC AD BC BD CD (there is no AA and no ACandCA) from the tree_arc_count list

            pairs_object = itertools.combinations(tree_arc_count, 2)
            for pair in pairs_object:
                pairs.append(pair)
            #print >> out, "Number of arcs combinations in this tree (", tree_wc, " words): ", len(pairs)  # for the test set I expect =COMBIN(86,2) = 3655, but I need combos within a sent only!
            #print pairs

            tples = 0
            for tple in pairs:
                tples += 1
                # print( tple[1][0], '/t', range(tple[0][0], tple[0][1]) )
                if tple[1][0] in range(tple[0][0], tple[0][1]) and not tple[1][1] in range(tple[0][0], tple[0][1]) and \
                                tple[1][0] != tple[0][0] and tple[1][1] != tple[0][1]:  # ([1,6],[2,7])
                    count += 1
                    in_sent +=1
                    for comment in comments:
                        print(comment)#.encode('utf-8')
                    print(tple)
                    print("====================================================")
                    # print('START of 2nd within the 1st arc', tple)
                if not tple[1][0] in range(tple[0][0], tple[0][1]) and tple[1][1] in range(tple[0][0], tple[0][1]) and \
                                tple[1][0] != tple[0][0] and tple[1][1] != tple[0][1]:  # ([1,6],[0,3])
                    count += 1
                    in_sent += 1
                    for comment in comments:
                        print(comment) #.encode('utf-8')
                    print(tple)
                    print("====================================================")
                    # print('END of of 2nd within the 1st arc', tple)
                else:
                    continue
            #print(tples)
            if in_sent > 0:
                nonpr_sents +=1
                #print in_sent
            if in_sent > 1:
                errors += 1
                #print 'it is likely that this sentence contains annotation errors'


    except:
        traceback.print_exc()
        print("\n\n ------- STATS MAY BE EMPTY OR INCOMPLETE ----------", file=sys.stderr)
        pass
    if args.stats:
        stats.print_basic_stats(out)

    print('non-projectivity: ', count) #number of intersections of arcs in the sent
    print('Ratio of non-projective dependencies : ', count / stats.word_count * 100)
    print('non-projective sentences: ', nonpr_sents)
    #print('In %d sentences there are more than 1 non-projective dependencies (annotation errors?)'%(errors))
    # find % of sents with non-projective deps and their ratio to total num of deps
    print('Ratio of sents with non-projective dependencies: ', nonpr_sents / stats.tree_count *100)










