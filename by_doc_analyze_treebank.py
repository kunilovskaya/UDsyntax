#!/usr/bin/python3

import itertools
from collections import OrderedDict

import numpy as np
from igraph import *


def nonprojectivity(tree):
    in_sent = 0  # resettable counter to mark that the current sent has a non-projective dependency
    nonprojectivesentence = False
    tree_arc_count = []  # list of all ID,HEAD pairs in a tree; it needs to be reset before processing every next tree
    for w in tree:  # w is a quadriplet dep, head, token, rel
        try:
            a = int(round(float(w[0]), 0))
            b = int(round(float(w[1]), 0))
            if a > b:
                tree_arc_count.append([b, a])
                # print( 'Reversed', '\t', [b,a] )
            else:
                tree_arc_count.append([a,
                                       b])
                # this is a list of all arcs defined as 2-member-lists of start-end elements
                # put in the right order: [[1, 2], [2, 18]]
        # print( 'Born straight', '\t', [a,b] )
        except ValueError:
            continue
            # print(w[0], w[1])
    # print >> out,'Lets inspect a random sample from the list of arcs: ',
    # tree_arc_count  # random.sample(self.tree_arc_count, 2))  # self.tree_arc_count[4: 10]

    pairs = []  # produce a list of all pairwise combinations of arcs
    # of type AB AC AD BC BD CD (there is no AA and no ACandCA) from the tree_arc_count list

    pairs_object = itertools.combinations(tree_arc_count, 2)
    for pair in pairs_object:
        pairs.append(pair)
    # print >> out, "Number of arcs combinations in this tree (", tree_wc, " words): ",
    # len(pairs)  # for the test set I expect =COMBIN(86,2) = 3655, but I need combos within a sent only!
    # print pairs

    tples = 0
    for tple in pairs:
        tples += 1
        # print( tple[1][0], '/t', range(tple[0][0], tple[0][1]) )
        if tple[1][0] in range(tple[0][0], tple[0][1]) and not tple[1][1] in range(tple[0][0], tple[0][1]) and \
                        tple[1][0] != tple[0][0] and tple[1][1] != tple[0][1]:  # ([1,6],[2,7])
            nonprojectivesentence = True
            in_sent += 1
            # print('START of 2nd within the 1st arc', tple)
        if not tple[1][0] in range(tple[0][0], tple[0][1]) and tple[1][1] in range(tple[0][0], tple[0][1]) and \
                        tple[1][0] != tple[0][0] and tple[1][1] != tple[0][1]:  # ([1,6],[0,3])
            nonprojectivesentence = True
            in_sent += 1
            # print('END of of 2nd within the 1st arc', tple)
        else:
            continue

    in_sent /= len(tree)

    return nonprojectivesentence, in_sent


def relation_distribution(tree):
    sent_relations = [w[3] for w in tree]
    distribution = {relation: sent_relations.count(relation) for relation in relations}

    # Converting to probability distribution
    # 'probabilities' are basically ratio of the rel in question to all rels in the sentence
    total = sum(distribution.values())  # counts the number of all instances of all dependancies in the sent

    for key in distribution:
        try:
            distribution[key] /= total
        except ZeroDivisionError:
            distribution[key] = 0
    return distribution  # a dict rel : its ratio to all words int the sent


def graph_metrics(tree):
    sentence_graph = Graph(len(tree) + 1)  # why is it called before filtering?
    if filtering:
        n_tokens = len([w for w in tree if w[3] != 'punct']) + 1
    else:
        n_tokens = len(tree) + 1
    sentence_graph = sentence_graph.as_directed()
    sentence_graph.vs["name"] = ['ROOT'] + [word[2] for word in tree]  # string of vertices' attributes called name
    sentence_graph.vs["label"] = sentence_graph.vs[
        "name"]  # the name attribute is renamed label, because in drawing vertex
    # labels are taken from the label attribute by default
    if filtering:
        edges = [(word[1], word[0]) for word in tree if word[3] != 'punct']  # (int(identifier), int(head), token, rel)
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
    # print(len(parts))
    # print(type(parts))
    if len(parts) == 1:
        nodes = [word[2] for word in tree if word[3] != 'punct' and word[3] != 'root']
        all_hds = []  # or a counter = 0?
        for node in nodes:
            hd = sentence_graph.shortest_paths_dijkstra('ROOT', node, mode=ALL)
            # print(node, hd[0][0])
            # print(type(hd[0][0]))# why the result is a two-level nested list?
            all_hds.append(hd[0][0])
        if all_hds:
            # this is a generic test for 'is everything ok?' which here takes the form of testing
            # whther the list is empty
            mhd = np.average(all_hds)
        else:
            mhd = None
    else:
        mhd = None
        # print('!!!')
        # print(tree)
        # gr_layout = sentence_graph.layout_kamada_kawai()
        # plot(sentence_graph, layout=gr_layout)

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
    for head_ind in range(len(s)):  # use s-index to refer to heads
        w = s[head_ind]
        if w[3] == 'root':  # we don't want -2 for each row containing root
            continue
        head_id = w[1]

        for dep_ind in range(len(s)):  # use s-index to refer to dependants
            w1 = s[dep_ind]
            if head_id == w1[0]:
                break
        dd = abs(head_ind - dep_ind) - 1
        # print(w[2], w1[2], dd, '\n')
        inbtw.append(abs(dd))

    # print(' '.join([w[2] for w in s]))
    #  why does this print an arbitrary num of sents and then the values for them (because it had file=sys.stderr!)
    # print(s)
    # print(len(inbtw))
    mdd = np.average(
        inbtw)
    # use this function instead of overt division of list sum by list length: if smth is wrong you'll get a warning!
    return mdd  # a list of mdd for each sentence


if __name__ == "__main__":
    many = sys.argv[1]  # always use final slash for folders
    folder = len(os.path.abspath(many).split('/')) - 1
    filtering = True  # Filter out punctuation and short sentences?
    min_length = 3
    '''
    I have thrown out unobserved relations and root
    there is no workaround the unordered nature of dicts, which gets me in trouble with several runs of the same script
    '''
    if filtering:
        relations = OrderedDict(
            [('acl', []), ('acl:relcl', []), ('advcl', []), ('advmod', []), ('amod', []), ('appos', []),
             ('aux', []), ('aux:pass', []), ('case', []), ('cc', []), ('ccomp', []), ('compound', []),
             ('conj', []), ('cop', []), ('csubj', []), ('csubj:pass', []), ('det', []),
             ('discourse', []), ('fixed', []), ('flat', []),
             ('flat:foreign', []), ('flat:name', []), ('iobj', []), ('mark', []), ('nmod', []),
             ('nsubj', []), ('nsubj:pass', []), ('nummod', []), ('nummod:gov', []),
             ('obj', []), ('obl', []), ('orphan', []), ('parataxis', []), ('xcomp', []), ])
    else:
        relations = OrderedDict([('acl', []), ('acl:relcl', []), ('advcl', []), ('advmod', []), ('amod', []),
                                 ('appos', []), ('aux', []), ('aux:pass', []), ('case', []), ('cc', []), ('ccomp', []),
                                 ('clf', []), ('compound', []), ('conj', []), ('cop', []), ('csubj', []), ('dep', []),
                                 ('det', []), ('discourse', []), ('dislocated', []), ('expl', []), ('fixed', []),
                                 ('flat', []), ('flat:foreign', []), ('flat:name', []), ('goeswith', []), ('iobj', []),
                                 ('list', []), ('mark', []), ('nmod', []), ('nsubj', []), ('nsubj:pass', []),
                                 ('nummod', []),
                                 ('nummod:entity', []), ('nummod:gov', []), ('obj', []), ('obl', []), ('obl:agent', []),
                                 ('orphan', []), ('parataxis', []), ('punct', []), ('reparandum', []), ('root', []),
                                 ('vocative', []), ('xcomp', []), ])

    metrics = OrderedDict([('Non-projective sentences', []),
                           ('Non-projectivity', []),
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

    # add doc and corpus names to the output
    headers = ['doc', 'group']

    for rel in relations.keys():
        headers.append(rel)
    for metric in metrics:
        headers.append(metric)
    for i in headers:
        print(i, end='\t')
    print('\n')

    arpack_options.maxiter = 3000

    # Now let's start analyzing the treebank as a set of documents
    bank = [f for f in os.listdir(many) if f.endswith('.conllu')]
    # and f.startswith('511329')]  # that's how to limit the input data to just offensive files
    # all_sents = 0  # or get them from supplied conllu-stats.py which works for onebig and for multiple input
    # good_sents = 0
    # all_good_words = 0

    for f in bank:
        # collect sentence-based counts
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
            current_sentence.append((int(identifier), int(head), token,
                                     rel))  # this is a list of data pieces from conllu that gets into [sentences]

        if current_sentence:
            sentences.append(current_sentence)
        # all_sents += len(sentences)

        if filtering:
            sentences = [s for s in sentences if len(s) >= min_length]
        # collect treebank-based averaged stats
        # for each new file clear values from the dicts
        for value in relations.values():
            del value[:]
        for value in metrics.values():
            del value[:]
        # sent_lengths=[] #list of sentence lengths for this doc/corpus

        for i in range(len(sentences)):  # why not for sentence in sentences:
            # if i % 1000 == 0:
            #     print('I have already analyzed %s sentences' % i, file=sys.stderr)
            sentence = sentences[i]
            # print(' '.join([w[2] for w in sentence]), file=sys.stderr)
            sgraph = graph_metrics(sentence)
            if not sgraph[7]:
                # if value fot the current sentence is None = sentence either has two or more components
                # or no content words other than those attached with root
                # print('YEs!')
                continue
            # count for actual corpus size in tokens after filtering, commented and blank lines are already excluded
            # all_good_words += len(sentence)
            # sent_lengths.append(len(sentence))
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
            metrics['Non-projectivity'].append(non_proj[1])

            rel_distribution = relation_distribution(sentence)
            for rel in relations.keys():
                relations[rel].append(rel_distribution[rel])
                # print(numrel, '\t', tokens)
            compre_diff = calculate_mdd(sentence)
            metrics['MDD'].append(compre_diff)

        '''
        count for all actually used sentences
        '''
        # good_sents += len(metrics['MDD'])

        doc = os.path.splitext(os.path.basename(many + f))[0]  # without extention
        cl = os.path.abspath(many).split('/')[folder]
        print(doc, cl, sep='\t', end='\t')
        # produce lists of average sent-Lengths for each doc to test significance of differences
        # print(doc, cl, np.mean(sent_lengths), len(metrics['MDD']), sum(sent_lengths), sep='\t', end='\n')
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

        # Uncomment to get revised stats o
        # folder = len(os.path.abspath(many).split('/')) - 1
        # print('Number of good sentences in ', os.path.abspath(many).split('/')[folder], '\t', good_sents)
        # print('Number of all sentences in ', os.path.abspath(many).split('/')[folder], '\t', all_sents)
        # print('Difference: ', all_sents - good_sents)
        # print('Corpus size after filtering: ', all_good_words)
