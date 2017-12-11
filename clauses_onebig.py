#!/usr/bin/python3
# extract dependencies and their pos from onebig
import os, sys
from os.path import basename
from itertools import permutations
from collections import OrderedDict
import numpy as np
from igraph import *
import matplotlib.pyplot as plt



onebig = sys.argv[1] # always use final slash for folders
doc = os.path.splitext(os.path.basename(onebig))[0]
filtering = True  # Filter out punctuation and short sentences?
min_length = 3
'''
I have thrown out unobserved relations and root
there is no workaround the unoedered nature of dicts, which gets me in trouble with several runs of the same script
'''

relations = OrderedDict([('acl', []), ('acl:relcl', []), ('advcl', []), ('advmod', []), ('amod', []), ('appos', []),
                             ('aux', []), ('aux:pass', []), ('case', []), ('cc', []), ('ccomp', []), ('compound', []),
                             ('conj', []), ('cop', []), ('dep', []), ('discourse', []), ('fixed', []), ('flat', []),
                             ('flat:foreign', []), ('flat:name', []), ('iobj', []), ('mark', []), ('nmod', []),
                             ('nsubj', []), ('nsubj:pass', []), ('nummod', []), ('nummod:entity', []), ('nummod:gov', []),
                             ('obj', []), ('obl', []), ('obl:agent', []), ('orphan', []), ('parataxis', []),
                            ('xcomp', []), ])

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

arpack_options.maxiter = 3000


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


#mycount all functions
#this counts number of occurences of short participles in Passive voice as roots without auxiliary
def allpass(septets):
    hasaux = False
    hasaux_nonroot = False
    ispass = False
    ispass_nonroot = False
    all_passsubj_fin = 0
    all_passsubj_spp = 0
    count_analitpass = 0
    count_analit_nonroot = 0
    count_morphpass = 0
    count_morphpass_nonroot = 0
    fin = 0

    for w in septets:
        if w[3] == 'root' and 'Variant=Short|VerbForm=Part|Voice=Pass' in w[6]:
            ispass = True
            id = w[0]
            for w1 in septets:
                if w1[1] == id and w1[3] != 'aux:pass':  # nsubj
                    c = deps.get(w1[3], 0)  # implicitly handles KeyError exception
                    c += 1
                    deps[w1[3]] = c
                    # print(w[4], '\t', w1[3])
                elif w1[1] == id and w1[3] == 'aux:pass':
                    hasaux = True
                    count_analitpass += 1
        if w[3] != 'root' and 'Variant=Short|VerbForm=Part|Voice=Pass' in w[6]:
            ispass_nonroot = True
            id = w[0]
            for w1 in septets:
                if w1[1] == id and w1[3] != 'aux:pass':  # nsubj
                    c = deps.get(w1[3], 0)  # implicitly handles KeyError exception
                    c += 1
                    deps[w1[3]] = c
                    # print(w[4], '\t', w1[3])
                elif w1[1] == id and w1[3] == 'aux:pass':
                    hasaux_nonroot = True
                    count_analit_nonroot += 1


        if 'VerbForm=Fin|Voice=Pass' in w[6]:
            fin += 1
            id = w[0]
            for w1 in septets:
                if w1[1] == id and w1[3] == 'nsubj:pass':
                    all_passsubj_fin +=1
        if 'Variant=Short|VerbForm=Part|Voice=Pass' in w[6]:
            id = w[0]
            for w1 in septets:
                if w1[1] == id and w1[3] == 'nsubj:pass':
                    all_passsubj_spp += 1

        # print(' '.join([w[2] for w in septets]), file=sys.stderr)
    if ispass_nonroot == True and hasaux_nonroot == False:
        count_morphpass_nonroot += 1

    if ispass == True and hasaux == False:
        count_morphpass += 1

    return count_analitpass, count_analit_nonroot, count_morphpass, count_morphpass_nonroot, fin, all_passsubj_fin, all_passsubj_spp

# Now let's start analyzing the treebank as one file
all_sents = 0  # or get them from supplied conllu-stats.py which works for onebig and for multiple input
good_sents = 0
all_good_words = 0
xcomp = 0


# inside = OrderedDict([('rootSPP_anal', []), ('nonrootSPP_anal', []),  ('rootSPP_morph', []), ('nonrootSPP_morph', []),
#                                                                  ('passFin', []), ('all_nsubjpass_fin', []), ('all_nsubjpass_spp', []),
# ])

# deps = {} # dict of types of passive verb dependants, except aux regardless root or non roots
# nsubj = {} # dict for types of nsubjpass
#
# print('corpus\troot+SPP+aux\t-root+SPP+aux\troot+SPP-aux\t-root+SPP-aux\tFinite\tall-nsubj_fin\tall_nsubj_spp')

words = open(onebig, 'r').readlines()
sentences = []
current_sentence = []  # определяем пустой список
for line in words:  # итерируем строки из обрабатываемого файла
    if line.strip() == '':  # что делать есть строка пустая:
        if current_sentence:
            sentences.append(current_sentence)
        current_sentence = []  # обнуляем список
        if len(sentences) % 1000 == 0:
            print('I have already read %s sentences' % len(sentences), file=sys.stderr)
        continue
    if line.strip().startswith('#'):
        continue
    res = line.strip().split('\t')
    (identifier, token, lemma, upos, xpos, feats, head, rel, misc1, misc2) = res
    if '.' in identifier:  # ignore empty nodes that are used in the enhanced representations
        continue
    current_sentence.append((int(identifier), int(head), token, rel, lemma, upos, feats))# this is a list of data pieces from conllu that gets into [sentences]

if current_sentence:
    sentences.append(current_sentence)
all_sents += len(sentences)


if filtering:
    sentences = [s for s in sentences if len(s) >= min_length]

# count_analitpass_nonroot = 0
# count_morphpass_nonroot = 0
# count = 0
# nodeps = 0
count_advcl = 0
advmarks = []
deps = []
arcs_counter = 0
for i in range(len(sentences)):  # why not for sentence in sentences:
    # ispass_nonroot = False
    # hasaux_nonroot = False
    sentence = sentences[i]
    # print(' '.join([w[2] for w in sentence]), file=sys.stderr)
    sgraph = graph_metrics(sentence)
    if not sgraph[7]:  # if value fot the current sentence is None = sentence either has two or more components or no content words other than those attached with root
        continue
    # count for actual corpus size in tokens after filtering, commented and blank lines are already excluded
    all_good_words += len(sentence)

    metrics['MHD'].append(sgraph[7])

    gotcha = False
    for w in sentence:

        arcs_counter += 1

        if w[3] == 'advcl' and 'VerbForm=Fin' in w[6]:
            id = w[0]
            for w1 in sentence:
                if w1[1] == id and 'nsubj' in w1[3]:
                    for w2 in sentence:
                        if w2[1]== id and 'mark' in w2[3]:
                            count_advcl +=1
                            advmarks.append(w2[4].strip())
                            print(w[2],'\t', w1[2], '\t', w2[2])
                            gotcha = True
                            freq_dic = {mark: [] for mark in advmarks}
                            for mark in set(advmarks):
                                freq_dic[mark] = advmarks.count(mark) / arcs_counter

    if gotcha == True:
        print(' '.join([w[2] for w in sentence]), file=sys.stderr)
print(count_advcl)
total = sum(freq_dic.values())
sorted = [(mark, freq_dic[mark]) for mark in sorted(freq_dic, key=freq_dic.get, reverse=True)]
for i in sorted[:20]:
    print(i)
print('Cumulative probability of all mark ', total)

# Uncomment (including counters and test-statements above) to run tests on the features and their values that the allpass function gets
#     for w in sentence:
#         if w[3] != 'root' and 'Variant=Short|VerbForm=Part|Voice=Pass' in w[6]:
#             ispass_nonroot = True
#             id = w[0]
#             deps_heads = [w1[1] for w1 in sentence]
#             print(deps_heads)
#             if not id in deps_heads:
#                 nodeps += 1
#                 print(deps_heads)
#             for w1 in sentence:
#
#                 if w1[1] == id and w1[3] != 'aux:pass':  # nsubj
#                     c = deps.get(w1[3], 0)  # implicitly handles KeyError exception
#                     c += 1
#                     deps[w1[3]] = c
#                     # print(w[4], '\t', w1[3])
#                 elif w1[1] == id and w1[3] == 'aux:pass':
#                     hasaux_nonroot = True
#                     count_analitpass_nonroot += 1
#                     print(' '.join([w[2] for w in sentence]), file=sys.stderr)
#
#             if ispass_nonroot == True and hasaux_nonroot == False:
#                 count_morphpass_nonroot += 1
#
# print(count_analitpass_nonroot, '\t', count_morphpass_nonroot, '\t', nodeps)
#
#     six_pass_types = allpass(sentence)
#     # print(six_pass_types)
#
#     inside['rootSPP_anal'].append(six_pass_types[0])
#     inside['nonrootSPP_anal'].append(six_pass_types[1])
#     inside['rootSPP_morph'].append(six_pass_types[2])
#     inside['nonrootSPP_morph'].append(six_pass_types[3])
#     inside['passFin'].append(six_pass_types[4])
#     inside['all_nsubjpass_fin'].append(six_pass_types[5])
#     inside['all_nsubjpass_spp'].append(six_pass_types[6])
#
#
#
# '''
# count for all actually used sentences
# '''
# good_sents += len(metrics['MHD'])
#
#
# print(doc, '\t', sum(inside['rootSPP_anal']), '\t',
#   sum(inside['nonrootSPP_anal']), '\t', sum(inside['rootSPP_morph']), '\t', sum(inside['nonrootSPP_morph']),
#   '\t', sum(inside['passFin']), '\t', sum(inside['all_nsubjpass_fin']),'\t', sum(inside['all_nsubjpass_spp']))
#
# print('passive verbs extends the following types of dependences in ', doc, ':','\t', deps, file=sys.stderr)
