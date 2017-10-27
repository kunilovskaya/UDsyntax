#!/usr/bin/python3

import os, sys
from itertools import permutations
from collections import OrderedDict
import numpy as np
from igraph import *
import matplotlib.pyplot as plt

many = sys.argv[1] # always use final slash for folders
folder = len(os.path.abspath(many).split('/')) - 1
filtering = True  # Filter out punctuation and short sentences?
min_length = 3
'''
I have thrown out unobserved relations
there is no workaround the unoedered nature of dicts, which gets me in trouble with several runs of the same script
'''
if filtering:
    relations = OrderedDict([('acl', []), ('acl:relcl', []), ('advcl', []), ('advmod', []), ('amod', []), ('appos', []),
                             ('aux', []), ('aux:pass', []), ('case', []), ('cc', []), ('ccomp', []), ('compound', []),
                             ('conj', []), ('cop', []), ('dep', []), ('discourse', []), ('fixed', []), ('flat', []),
                             ('flat:foreign', []), ('flat:name', []), ('iobj', []), ('mark', []), ('nmod', []),
                             ('nsubj', []), ('nsubj:pass', []), ('nummod', []), ('nummod:entity', []), ('nummod:gov', []),
                             ('obj', []), ('obl', []), ('obl:agent', []), ('orphan', []), ('parataxis', []),
                             ('root', []), ('xcomp', []), ])
else:
    relations = OrderedDict([('acl', []), ('acl:relcl', []), ('advcl', []), ('advmod', []), ('amod', []),
                             ('appos', []), ('aux', []), ('aux:pass', []), ('case', []), ('cc', []), ('ccomp', []),
                             ('clf', []), ('compound', []), ('conj', []), ('cop', []), ('csubj', []), ('dep', []),
                             ('det', []), ('discourse', []), ('dislocated', []), ('expl', []), ('fixed', []),
                             ('flat', []), ('flat:foreign', []), ('flat:name', []), ('goeswith', []), ('iobj', []),
                             ('list', []), ('mark', []), ('nmod', []), ('nsubj', []), ('nsubj:pass', []), ('nummod', []),
                             ('nummod:entity', []), ('nummod:gov', []), ('obj', []), ('obl', []), ('obl:agent', []),
                             ('orphan', []), ('parataxis', []), ('punct', []), ('reparandum', []), ('root', []),
                             ('vocative', []), ('xcomp', []), ])

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

#add doc and corpus names to the output
headers = ['doc', 'class']

for rel in relations.keys():
    headers.append(rel)
for metric in metrics:
    headers.append(metric)
for i in headers:
    print(i, end='\t')
print('\n')

arpack_options.maxiter = 3000

filtering = True  # Filter out punctuation and short sentences?
min_length = 3

def nonprojectivity(tree):
    if filtering:
        arcs = len([w for w in tree if w[3] != 'punct'])
    else:
        arcs = len(tree)
    n_arcs = 0
    nonprojectivesentence = False
    pairs = permutations(tree, 2)
    for pair in pairs:
        (dep0, head0, token0, rel0) = pair[0]
        (dep1, head1, token1, rel1) = pair[1]
        if filtering:
            if rel0 == 'punct' or rel1 == 'punct':
                continue
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
        try:
            distribution[key] /= total
        except ZeroDivisionError:
            distribution[key] = 0
    return distribution


def graph_metrics(tree):
    sentence_graph = Graph(len(tree) + 1)
    if filtering:
        n_tokens = len([w for w in tree if w[3] != 'punct']) + 1
    else:
        n_tokens = len(tree) + 1
    sentence_graph = sentence_graph.as_directed()
    sentence_graph.vs["name"] = ['ROOT'] + [word[2] for word in tree]
    sentence_graph.vs["label"] = sentence_graph.vs["name"]
    if filtering:
        edges = [(word[1], word[0]) for word in tree if word[3] != 'punct']
    else:
        edges = [(word[1], word[0]) for word in tree]
    # print([w[0] for w in tree])
    # print(edges)
    sentence_graph.add_edges(edges)
    sentence_graph.vs.find("ROOT")["shape"] = 'diamond'
    sentence_graph.vs.find("ROOT")["size"] = 40
    if filtering:
        disconnected = [vertex.index for vertex in sentence_graph.vs if vertex.degree() == 0]
        sentence_graph.delete_vertices(disconnected)

    av_degree = np.average(sentence_graph.degree(type="out"))
    max_degree = max(sentence_graph.degree(type="out"))

    try:
        communities = sentence_graph.community_leading_eigenvector()
    except InternalError:
        communities = ['dummy']

    comm_size = n_tokens / len(communities)

    av_path_length = sentence_graph.average_path_length()

    density = sentence_graph.density()
    diameter = sentence_graph.diameter()

    # Uncomment to produce visuals DONT ever uncomment or pictures never stop popping up!!!
    # if 0.9 < av_degree < 0.98 and len(tree) > 4:
    #     if True:
            # print(' '.join([w[2] for w in tree]), file=sys.stderr)
            # print(av_degree, file=sys.stderr)
            # gr_layout = sentence_graph.layout_kamada_kawai()
            # plot(communities, layout=gr_layout)
    return av_degree, max_degree, len(communities), comm_size, av_path_length, density, diameter

# Now let's start analyzing the treebank as a set of documents
bank = [f for f in os.listdir(many) if f.endswith('.conllu')]
for f in bank:
    #collect sentence-based counts
    words = open(many + f, 'r').readlines()
    sentences = []
    current_sentence = []  # определяем пустой список
    for line in words:  # итерируем строки из обрабатываемого файла
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
        if '.' in identifier:  # ignore empty nodes that are used in the enhanced representations
            continue
        current_sentence.append((int(identifier), int(head), token, rel))

    if current_sentence:
        sentences.append(current_sentence)

    if filtering:
        sentences = [s for s in sentences if len(s) >= min_length]
    # collect treebank-based averaged stats
    # for each new file clear values from the dicts
    for value in relations.values():
        del value[:]
    for value in metrics.values():
        del value[:]
    for i in range(len(sentences)):  # why not for sentence in sentences:
        # if i % 1000 == 0:
        #     print('I have already analyzed %s sentences' % i, file=sys.stderr)
        sentence = sentences[i]
        # print(' '.join([w[2] for w in sentence]), file=sys.stderr)
        non_proj = nonprojectivity(sentence)

        metrics['Non-projective sentences'].append(non_proj[0])
        metrics['Non-projective arcs'].append(non_proj[1])

        rel_distribution = relation_distribution(sentence)

        for rel in relations.keys():
            relations[rel].append(rel_distribution[rel])
        sgraph = graph_metrics(sentence)
        metrics['Average out-degree'].append(sgraph[0])
        metrics['Max out-degree'].append(sgraph[1])
        metrics['Number of communities'].append(sgraph[2])
        metrics['Average community size'].append(sgraph[3])
        metrics['Average path length'].append(sgraph[4])
        metrics['Density'].append(sgraph[5])
        metrics['Diameter'].append(sgraph[6])

    doc = os.path.splitext(os.path.basename(many+f))[0]#without extention
    cl = os.path.abspath(many).split('/')[folder]
    print(doc, cl, sep='\t', end='\t')
    '''
    this is needed to avoid printing a new line after each file or printing all files to one line
    '''
    allvalues = []
    for rel in relations.keys():
        data = np.average(relations[rel])  # pulls out lists of values and averages them for sents in this bank
        allvalues.append(str(data))
    for metric in metrics:
        data = np.average(metrics[metric])
        allvalues.append(str(data))
    print('\t'.join(allvalues))


