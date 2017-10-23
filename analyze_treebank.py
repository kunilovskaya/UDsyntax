#!/usr/bin/python3
import fileinput
from itertools import permutations
from collections import Counter
import numpy as np
import sys


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


relations = 'nsubj obj iobj csubj ccomp xcomp obl vocative expl dislocated advcl advmod discourse aux cop mark nmod ' \
            'appos nummod acl amod det clf case conj cc fixed flat compound list parataxis orphan goeswith ' \
            'reparandum punct root dep'.split()

relations_counter = Counter({rel: 0 for rel in relations})  # Here will go probabilities of arc labels

sentences = []

current_sentence = []
for line in fileinput.input():
    if line.strip() == '':
        if current_sentence:
            sentences.append(current_sentence)
        current_sentence = []
        if len(sentences) % 100 == 0:
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

for i in range(len(sentences)):
    if i % 100 == 0:
        print('I have already analyzed %s sentences' % i, file=sys.stderr)

    sentence = sentences[i]
    # print(' '.join([w[2] for w in sentence]), file=sys.stderr)
    non_proj = nonprojectivity(sentence)
    nonprojectivities.append(non_proj[0])
    non_arcs.append(non_proj[1])
    sent_relations = [w[3] for w in sentence]
    rel_distribution = Counter({rel: sent_relations.count(rel) for rel in relations})

    # Converting to probability distribution
    total = sum(rel_distribution.values())
    for key in rel_distribution:
        rel_distribution[key] /= total

    relations_counter = relations_counter + rel_distribution

print('Non-projective sentences:', np.average(nonprojectivities))
print('Non-projective arcs:', np.average(non_arcs))

# Averaging probabilities of arc labels
for key in relations_counter:
    relations_counter[key] /= len(sentences)

for rel in sorted(relations):
    print(rel, relations_counter[rel])
