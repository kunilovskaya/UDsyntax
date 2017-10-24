#!/usr/bin/python3
import fileinput
from itertools import permutations
from collections import OrderedDict
import numpy as np
from igraph import *

arpack_options.maxiter = 3000


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
    # 'probabilities' are basically ratio of the rel in question to all rels in the sentence
    total = sum(distribution.values())
    for key in distribution:
        distribution[key] /= total
    return distribution


def graph_metrics(tree):
    sentence_graph = Graph(len(tree) + 1)
    sentence_graph = sentence_graph.as_directed()
    sentence_graph.vs["name"] = ['ROOT'] + [word[2] for word in tree]
    sentence_graph.vs["label"] = sentence_graph.vs["name"]
    edges = [(word[1], word[0]) for word in tree]
    sentence_graph.add_edges(edges)
    sentence_graph.vs.find("ROOT")["shape"] = 'diamond'
    sentence_graph.vs.find("ROOT")["size"] = 40

    av_degree = np.average(sentence_graph.degree(type="out"))
    max_degree = max(sentence_graph.degree(type="out"))

    try:
        communities = sentence_graph.community_leading_eigenvector()
    except InternalError:
        communities = ['dummy']

    comm_size = len(tree)/len(communities)

    av_path_length = sentence_graph.average_path_length()

    density = sentence_graph.density()
    diameter = sentence_graph.diameter()

    # layout = sentence_graph.layout_kamada_kawai()
    # plot(communities, layout=layout)
    return av_degree, max_degree, len(communities), comm_size, av_path_length, density, diameter


relations = 'nsubj obj iobj csubj ccomp xcomp obl vocative expl dislocated advcl advmod discourse aux cop mark nmod ' \
            'appos nummod acl amod det clf case conj cc fixed flat compound list parataxis orphan goeswith ' \
            'reparandum punct root dep'.split()

relations = {rel: [] for rel in relations}  # Here will go probabilities of arc labels

# Now let's start analyzing the treebank

sentences = []

current_sentence = []  # определяем пустой список
for line in fileinput.input():  # итерируем строки из обрабатываемого файла
    if line.strip() == '':  # что делать есть строка пустая:
        if current_sentence:  # и при этом в списке уже что-то записано
            # то добавляем в другой список sentences содержимое списка current_sentences
            sentences.append(current_sentence)
        current_sentence = []  # обнуляем список

        # if the number of sents can by devided by 1K without a remainder.
        # В этом случае, т.е. после каждого 1000-ного предложения печатай месседж. Удобно!
        if len(sentences) % 1000 == 0:
            print('I have already read %s sentences' % len(sentences), file=sys.stderr)
        continue
    if line.strip().startswith('#'):
        continue
    res = line.strip().split('\t')
    (identifier, token, lemma, upos, xpos, feats, head, rel, misc1, misc2) = res
    if '.' in identifier:  # ignore empty nodes possible in the enhanced representations
        continue
    # во всех остальных случаях имеем дело со строкой по отдельному слову
    current_sentence.append((int(identifier), int(head), token, rel))

if current_sentence:
    sentences.append(current_sentence)

metrics = OrderedDict([('Non-projective sentences', []),
                       ('Non-projective arcs', []),
                       ('Average out-degree', []),
                       ('Max out-degree', []),
                       ('Number of communities', []),
                       ('Average community size', []),
                       ('Average path length', []),
                       ('Density', []),
                       ('Diameter', []),
                       ])

for i in range(len(sentences)):  # why not for sentence in sentences:
    if i % 1000 == 0:
        print('I have already analyzed %s sentences' % i, file=sys.stderr)

    sentence = sentences[i]
    # print(' '.join([w[2] for w in sentence]), file=sys.stderr)
    non_proj = nonprojectivity(sentence)
    metrics['Non-projective sentences'].append(non_proj[0])
    metrics['Non-projective arcs'].append(non_proj[1])

    rel_distribution = relation_distribution(sentence)
    for rel in relations:
        relations[rel].append(rel_distribution[rel])

    sgraph = graph_metrics(sentence)
    metrics['Average out-degree'].append(sgraph[0])
    metrics['Max out-degree'].append(sgraph[1])
    metrics['Number of communities'].append(sgraph[2])
    metrics['Average community size'].append(sgraph[3])
    metrics['Average path length'].append(sgraph[4])
    metrics['Density'].append(sgraph[5])
    metrics['Diameter'].append(sgraph[6])

print('Feature\tAverage\tDeviation\tObservations')
for metric in metrics:
    print(metric, '\t', np.average(metrics[metric]), '\t', np.std(metrics[metric]), '\t', len(metrics[metric]))

for rel in sorted(relations.keys()):
    data = relations[rel]
    print(rel + '\t', np.average(data), '\t', np.std(data), '\t', len(data))
