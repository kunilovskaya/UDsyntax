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
                            ('MHD', []),
                            ('MDD', []),
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
    sentence_graph = Graph(len(tree) + 1) #why is it called before filtering?
    if filtering:
        n_tokens = len([w for w in tree if w[3] != 'punct']) + 1
    else:
        n_tokens = len(tree) + 1
    sentence_graph = sentence_graph.as_directed()
    sentence_graph.vs["name"] = ['ROOT'] + [word[2] for word in tree] #string of vertices' attributes called name
    sentence_graph.vs["label"] = sentence_graph.vs["name"] #the name attribute is renamed label, because in drawing vertex labels are taken from the label attribute by default
    if filtering:
        edges = [(word[1], word[0]) for word in tree if word[3] != 'punct'] #(int(identifier), int(head), token, rel)
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

    '''
        calculate processing difficulty=mean hierachical distance(MHD) as average value of all path lengths
        traveling from the root to all nodes along the dependency edges (Jing, Liu 2015 : 164)
    '''
    parts = sentence_graph.components(mode=WEAK)
    #print(len(parts))
    # print(type(parts))
    if len(parts) == 1:
        nodes = [word[2] for word in tree if word[3] != 'punct' and word[3] != 'root']
        all_hds = []  # or a counter = 0?
        for node in nodes:
            hd = sentence_graph.shortest_paths_dijkstra('ROOT', node, mode=ALL)
            # print(node, hd[0][0])
            # print(type(hd[0][0]))# why the result is a two-level nested list?
            all_hds.append(hd[0][0])
        if all_hds:  # this is a generic test for 'is everything ok?' which here takes the form of testing whther the list is empty
            mhd = np.average(all_hds)
        else:
            mhd = None
    else:
        mhd = None
        #print('!!!')
        #print(tree)
        #gr_layout = sentence_graph.layout_kamada_kawai()
        #plot(sentence_graph, layout=gr_layout)

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
    return av_degree, max_degree, len(communities), comm_size, av_path_length, density, diameter, mhd

def calculate_mdd(tree):
    """
     calculate comprehension difficulty=mean dependency distance(MDD) as “the distance between words and their parents,
     measured in terms of intervening words.” (Hudson 1995 : 16)
    """
    s = [q for q in tree if q[3] != 'punct']  # that's an elegant way to create a new list of sentence quadriplets! without repeating any code
    inbtw = []
    for head_ind in range(len(s)):  # use s-index to refer to heads
        w = s[head_ind]
        if w[3] == 'root':  # we don't want -2 for each row containing root
            continue
        head = w[1]

        for dep_ind in range(len(s)):  # use s-index to refer to dependants
            w1 = s[dep_ind]
            if head == w1[0]:
                break
        dd = abs(head_ind - dep_ind) - 1
        # print(w[2], w1[2], dd, '\n')
        inbtw.append(abs(dd))

    # print(' '.join([w[2] for w in s])) # why does this print an arbitrary num of sents and then the values for them (because it had file=sys.stderr!)
    # print(s)
    # print(len(inbtw))
    mdd = np.average(inbtw)  # use this function instead of overt division of list sum by list length: if smth is wrong you'll get a warning!
    return mdd  # a list of mdd for each sentence

# Now let's start analyzing the treebank as a set of documents
bank = [f for f in os.listdir(many) if f.endswith('.conllu')]  # and f.startswith('511329')]  # that's how to limit the input data to just offensive files
all_sents = 0  # or get them from supplied conllu-stats.py which works for onebig and for multiple input
good_sents = 0
all_good_words = 0
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
        current_sentence.append((int(identifier), int(head), token, rel))# this is a list of data pieces from conllu that gets into [sentences]

    if current_sentence:
        sentences.append(current_sentence)
    all_sents += len(sentences)


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
        sgraph = graph_metrics(sentence)
        if not sgraph[7]:  # if value fot the current sentence is None = sentence either has two or more components or no content words other than those attached with root
            # print('YEs!')
            continue
        # count for actual corpus size in tokens after filtering, commented and blank lines are already excluded
        all_good_words += len(sentence)
        metrics['Average out-degree'].append(sgraph[0])
        metrics['Max out-degree'].append(sgraph[1])
        metrics['Number of communities'].append(sgraph[2])
        metrics['Average community size'].append(sgraph[3])
        metrics['Average path length'].append(sgraph[4])
        metrics['Density'].append(sgraph[5])
        metrics['Diameter'].append(sgraph[6])
        metrics['MHD'].append(sgraph[7])
        non_proj = nonprojectivity(sentence)

        metrics['Non-projective sentences'].append(non_proj[0])
        metrics['Non-projective arcs'].append(non_proj[1])

        rel_distribution = relation_distribution(sentence)

        for rel in relations.keys():
            relations[rel].append(rel_distribution[rel])

        compre_diff = calculate_mdd(sentence)
        metrics['MDD'].append(compre_diff)

    '''
    count for all actually used sentences
    '''
    good_sents += len(metrics['MDD'])

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
    # print(metrics)
    for metric in metrics:
        data = np.average(metrics[metric])
        allvalues.append(str(data))
    print('\t'.join(allvalues))

## Uncomment to get revised stats o
# folder = len(os.path.abspath(many).split('/')) - 1
# print('Number of good sentences in ', os.path.abspath(many).split('/')[folder], '\t', good_sents)
# print('Number of all sentences in ', os.path.abspath(many).split('/')[folder], '\t', all_sents)
# print('Difference: ', all_sents - good_sents)
# print('Corpus size after filtering: ', all_good_words)
