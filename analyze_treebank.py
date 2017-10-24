#!/usr/bin/python3
import fileinput
import sys
from itertools import permutations
from igraph import *
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

def graph_metrics(tree):
    sentence_graph = Graph(len(tree)+1)
    sentence_graph = sentence_graph.as_directed()
    sentence_graph.vs["name"] = ['ROOT']+[word[2] for word in tree]
    sentence_graph.vs["label"] = sentence_graph.vs["name"]
    edges = [(int(word[1]), int(word[0])) for word in tree]
    sentence_graph.add_edges(edges)
    sentence_graph.vs.find("ROOT")["color"]= 'green'

    # layout = sentence_graph.layout_kamada_kawai()
    # plot(sentence_graph, layout=layout)

    return sentence_graph



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

nonprojectivities = []
non_arcs = []
average_degrees = []
max_degrees = []

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

    sgraph = graph_metrics(sentence)
    average_degrees.append(np.average(sgraph.degree(type="out")))
    max_degrees.append(max(sgraph.degree(type="out")))

print('Feature\tAverage\tDeviation\tObservations')
print('Non-projective sentences\t', np.average(nonprojectivities), '\t', np.std(nonprojectivities), '\t',
      len(nonprojectivities))
print('Non-projective arcs\t', np.average(non_arcs), '\t', np.std(non_arcs), '\t', len(non_arcs))
print('Average out-degree\t', np.average(average_degrees), '\t', np.std(average_degrees), '\t', len(average_degrees))
print('Max out-degree\t', np.average(max_degrees), '\t', np.std(max_degrees), '\t', len(max_degrees))

for rel in sorted(relations.keys()):
    data = relations[rel]
    print(rel + '\t', np.average(data), '\t', np.std(data), '\t', len(data))
