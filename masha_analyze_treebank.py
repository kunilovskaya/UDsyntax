#!/usr/bin/python3
import fileinput
import sys
from itertools import permutations

import numpy as np


def relation_distribution(tree):
    sent_relations = [w[3] for w in tree] #w[3] in {sentences} is rel; this creates a list of all instances of all rel in the tree
    distribution = {rel: sent_relations.count(rel) for rel in relations} # create a dic with rels from the pre-defined list below for keys and a count for each from this sentence

    # Converting to probability distribution
    total = sum(distribution.values()) # total number of all rels in a sent
    for key in distribution:
        distribution[key] /= total # this converts counts into relative 'probabilities' that are basically ratio of the rel in question to all rels in the sentence
    return distribution # a dic {rel:probability} for this sentence


relations = 'nsubj obj iobj csubj ccomp xcomp obl vocative expl dislocated advcl advmod discourse aux cop mark nmod ' \
            'appos nummod acl amod det clf case conj cc fixed flat compound list parataxis orphan goeswith ' \
            'reparandum punct root dep'.split()

relations = {rel: [] for rel in relations}  # Here will go probabilities of arc labels

sentences = []

current_sentence = [] # определяем пустой список
for line in fileinput.input(): # итерируем строки из обрабатываемого файла
    if line.strip() == '': # что делать есть строка пустая:
        if current_sentence: #  и при этом в списке уже что-то записано
            sentences.append(current_sentence) # то добавляем в другой список sentences содержимое списка current_sentences
        current_sentence = [] # обнуляем список
        if len(sentences) % 1000 == 0: # if the number of sents can by devided by 1K without a remainder. В этом случае, т.е. после каждого 1000-ного предложения печатай месседж. Удобно!
            print('I have already read %s sentences' % len(sentences), file=sys.stderr)
        continue
    if line.strip().startswith('#'):
        continue
    res = line.strip().split('\t')
    (identifier, token, lemma, upos, xpos, feats, head, rel, misc1, misc2) = res
    if '.' in identifier: # ignore empty nodes possible in the enhanced representations
        continue
    
    current_sentence.append((float(identifier), float(head), token, rel)) # во всех остальных случаях имеем дело со строкой по отдельному слову

if current_sentence:
    sentences.append(current_sentence)

non_arcs = []

for i in range(len(sentences)): # why not for sentence in sentences:
    if i % 1000 == 0:
        print('I have already analyzed %s sentences' % i, file=sys.stderr)

    sentence = sentences[i]
    # print(' '.join([w[2] for w in sentence]), file=sys.stderr)

    rel_distribution = relation_distribution(sentence) # function call; w[3] in {sentences} is rel
    for rel in relations:
        relations[rel].append(rel_distribution[rel]) #a dic with lists as values; the lists collects the probabilitites of each rel for each sentence

print('Feature\tAverage\tDeviation\tObservations')

for rel in sorted(relations.keys()): # iterates over the dic, stores lists of probabilities into the 'data' variable and applies numpy methods to these lists
    data = relations[rel]
    print(rel + '\t', np.average(data), '\t', np.std(data), '\t', len(data))

