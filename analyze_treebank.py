#!/usr/bin/python3
import fileinput
import sys
from itertools import permutations

import numpy as np


def nonprojectivity(tree):
    arcs = len(tree)
    n_arcs = 0
    nonprojectivesentence = False
    pairs = permutations(tree, 2)
    for pair in pairs:
        (dep0, head0, token0, rel0) = pair[0]
        (dep1, head1, token1, rel1) = pair[1]
        if dep0 < dep1 < head0 < head1 or head1 < head0 < dep1 < dep0 or dep1 < head0 < head1 < dep0 \
                or dep0 < head1 < head0 < dep1 or head0 < dep1 < dep0 < head1 or head1 < dep0 < dep1 < head0:
            nonprojectivesentence = True
            n_arcs += 2
    nonprojective_ratio = n_arcs / arcs
    return nonprojectivesentence, nonprojective_ratio


def relation_distribution(tree):
    sent_relations = [w[3] for w in tree]
    distribution = {rel: sent_relations.count(rel) for rel in relations}

    # Converting to probability distribution
    total = sum(distribution.values())
    for key in distribution:
        distribution[key] /= total # 'probabilities' are basically ratio of the rel in question to all rels in the sentence
    return distribution


relations = 'nsubj obj iobj csubj ccomp xcomp obl vocative expl dislocated advcl advmod discourse aux cop mark nmod ' \
            'appos nummod acl amod det clf case conj cc fixed flat compound list parataxis orphan goeswith ' \
            'reparandum punct root dep'.split()

relations = {rel: [] for rel in relations}  # Here will go probabilities of arc labels

sentences = []

current_sentence = []
for line in fileinput.input():
    if line.strip() == '': #то же что line = line.strip()?
        if current_sentence: # if file has not ended? 
            sentences.append(current_sentence)
        current_sentence = []
        if len(sentences) % 1000 == 0: # if the number of sentences can by devided by 1K without a remainder. Why?
            print('I have already read %s sentences' % len(sentences), file=sys.stderr)
        continue
    if line.strip().startswith('#'):
        continue
    res = line.strip().split('\t')
    (identifier, token, lemma, upos, xpos, feats, head, rel, misc1, misc2) = res
    if '.' in identifier:
        continue
    current_sentence.append((float(identifier), float(head), token, rel))

if current_sentence:
    sentences.append(current_sentence)

nonprojectivities = []
non_arcs = []

for i in range(len(sentences)): # why not for sentence in sentences:
    if i % 1000 == 0:
        print('I have already analyzed %s sentences' % i, file=sys.stderr)

    sentence = sentences[i]
    # print(' '.join([w[2] for w in sentence]), file=sys.stderr)
    non_proj = nonprojectivity(sentence)
    nonprojectivities.append(non_proj[0])
    non_arcs.append(non_proj[1])

    rel_distribution = relation_distribution(sentence)
    for rel in relations:
        relations[rel].append(rel_distribution[rel])

print('Feature\tAverage\tDeviation\tObservations')
print('Non-projective sentences\t', np.average(nonprojectivities), '\t', np.std(nonprojectivities), '\t',
      len(nonprojectivities))
print('Non-projective arcs\t', np.average(non_arcs), '\t', np.std(non_arcs), '\t', len(non_arcs))

for rel in sorted(relations.keys()):
    data = relations[rel]
    print(rel + '\t', np.average(data), '\t', np.std(data), '\t', len(data))
