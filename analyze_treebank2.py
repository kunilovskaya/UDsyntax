#!/usr/bin/python3
#

import fileinput
from collections import OrderedDict
from itertools import permutations
import numpy as np
from igraph import *


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
        if rel0 == 'root' or rel1 == 'root':
            continue
        if filtering:
            if rel0 == 'punct' or rel1 == 'punct':
                continue
        if dep0 < dep1 < head0 < head1 or head1 < head0 < dep1 < dep0 or dep1 < head0 < head1 < dep0 \
                or dep0 < head1 < head0 < dep1 or head0 < dep1 < dep0 < head1 or head1 < dep0 < dep1 < head0:
            nonprojectivesentence = True
            n_arcs += 2
    nonprojective_ratio = n_arcs / arcs
    return nonprojectivesentence, nonprojective_ratio


def relation_distribution(tree, i_relations):
    sent_relations = [w[3] for w in tree]
    distribution = {rel: sent_relations.count(rel) for rel in i_relations}

    # Converting to probability distribution
    # 'probabilities' are basically ratio of the rel in question to all rels in the sentence
    total = sum(distribution.values())
    if total:  # Only if there are any interesting relations in the sentence
        for key in distribution:
            distribution[key] /= total
    return distribution


def graph_metrics(tree):
    sentence_graph = Graph(len(tree) + 1)  # why is it called before filtering?
    if filtering:
        n_tokens = len([w for w in tree if w[3] != 'punct']) + 1
    else:
        n_tokens = len(tree) + 1
    sentence_graph = sentence_graph.as_directed()
    sentence_graph.vs["name"] = ['ROOT'] + [word[2] for word in tree]  # string of vertices' attributes called name
    # the name attribute is renamed label,
    # because in drawing vertex labels are taken from the label attribute by default
    sentence_graph.vs["label"] = sentence_graph.vs["name"]
    if filtering:
        edges = [(word[1], word[0]) for word in tree if word[3] != 'punct']  # (int(identifier), int(head), token, rel)
    else:
        edges = [(word[1], word[0]) for word in tree]
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
    mhd = None
    if len(parts) == 1:
        nodes = [word[2] for word in tree if word[3] != 'punct' and word[3] != 'root']
        all_hds = []  # or a counter = 0?
        for node in nodes:
            hd = sentence_graph.shortest_paths_dijkstra('ROOT', node, mode=ALL)
            all_hds.append(hd[0][0])
        if all_hds:
            # this is a generic test for 'is everything ok?' which here takes the form of testing
            # whether the list is empty
            mhd = np.average(all_hds)

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

    s = [q for q in tree if q[3] != 'punct']
    # that's an elegant way to create a new list of sentence quadriplets! without repeating any code
    inbtw = []
    if len(s) > 1:
        for s_word_id in range(len(s)):  # use s-index to refer to words
            w = s[s_word_id]
            head_id = w[1]
            if head_id == 0:
                continue
            s_head_id = None
            for s_word_id_2 in range(len(s)):  # use s-index to refer to heads
                w1 = s[s_word_id_2]
                if head_id == w1[0]:
                    s_head_id = s_word_id_2
                    break
            dd = abs(s_word_id - s_head_id) - 1
            inbtw.append(dd)
    # use this function instead of overt division of list sum by list length: if smth is wrong you'll get a warning!
    mdd = np.average(inbtw)
    return mdd  # average MDD for the sentence


if __name__ == "__main__":
    arpack_options.maxiter = 3000

    filtering = True  # Filter out punctuation and short sentences?
    min_length = 3

    modes = ['overview', 'perfile', 'persentence']

    mode = modes[0]

    if filtering:
        relations = "acl acl:relcl advcl advmod amod appos aux aux:pass case cc ccomp compound conj cop csubj " \
                    "csubj:pass det discourse fixed flat flat:foreign flat:name iobj mark nmod nsubj nsubj:pass " \
                    "nummod nummod:gov obj obl orphan parataxis xcomp".split()
    else:
        relations = 'nsubj obj iobj csubj ccomp xcomp obl vocative expl dislocated advcl advmod discourse aux cop ' \
                    'mark nmod appos nummod acl amod det clf case conj cc fixed flat compound list parataxis orphan ' \
                    'goeswith reparandum punct root dep acl:relcl flat:name nsubj:pass nummod:gov aux:pass ' \
                    'flat:foreign obl:agent nummod:entity'.split()

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

            # if the number of sents can by divided by 1K without a remainder.
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
                           ('MHD', []),
                           ('MDD', []),
                           ])
    if filtering:
        sentences = [s for s in sentences if len(s) >= min_length]

    for i in range(len(sentences)):  # why not for sentence in sentences:
        if i % 1000 == 0:
            print('I have already analyzed %s sentences' % i, file=sys.stderr)

        sentence = sentences[i]
        # print(' '.join([w[2] for w in sentence]), file=sys.stderr)
        sgraph = graph_metrics(sentence)
        if not sgraph[7]:
            continue
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

        rel_distribution = relation_distribution(sentence, relations)
        for rel in relations.keys():
            relations[rel].append(rel_distribution[rel])
        compre_diff = calculate_mdd(sentence)
        metrics['MDD'].append(compre_diff)

    if mode == 'persentence':
        print('Sentence\t', '\t'.join(metrics.keys()) + '\t', '\t'.join(sorted(relations.keys())), '\tClass')
        for sent in range(len(sentences)):
            print(sent, '\t', end=' ')
            for metric in metrics:
                print(metrics[metric][sent], end='\t')
            for rel in sorted(relations.keys()):
                data = relations[rel][sent]
                print(data, end='\t')

            class_label = fileinput.filename().split('/')[0]  # the directory
            print(class_label)

    elif mode == 'perfile':
        # print('File\t', '\t'.join(metrics.keys())+'\t', '\t'.join(sorted(relations.keys())), '\tClass')
        print(fileinput.filename().split('/')[1].split('.')[0] + '\t', end=' ')
        for metric in metrics:
            print(np.average(metrics[metric]), end='\t')
        for rel in sorted(relations.keys()):
            data = np.average(relations[rel])
            print(data, end='\t')

        class_label = fileinput.filename().split('/')[0]  # the directory
        print(class_label)

    elif mode == 'overview':
        print('Feature\tAverage\tDeviation\tObservations')
        for metric in metrics:
            print(metric, '\t', np.average(metrics[metric]), '\t', np.std(metrics[metric]), '\t', len(metrics[metric]))
            # plot metric histogram if needed
            # plt.hist(metrics[metric], 50)
            # plt.grid(True)
            # plt.title(metric)
            # plt.show()
        for rel in sorted(relations.keys()):
            data = relations[rel]
            print(rel + '\t', np.average(data), '\t', np.std(data), '\t', len(data))
