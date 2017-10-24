#!/usr/bin/python3
#import fileinput
from os.path import basename
import sys, codecs

input = sys.argv[1]

trees = codecs.open(input, 'r', 'utf-8').readlines()

allsubj = 0
nsubj_nom = 0
nom_lems = []
nsubj_dat = 0
dat_lems = []
other = 0
allrels = 0

#sent_data = [] #has a list of
for line in trees: #fileinput.input(): # итерируем строки из обрабатываемого файла
    #print(line)
    if line.strip().startswith('# text ='):
        #raw_sents.append(line)
        continue
    if line.strip().startswith('#') or line.strip() == '':
        continue

    res = line.strip().split('\t')
    (identifier, token, lemma, upos, xpos, feats, head, rel, misc1, misc2) = res
    allrels +=1
    if 'nsubj' in rel:
        allsubj +=1
        if 'Case=Nom' in feats: # ignore empty nodes possible in the enhanced representations
            nsubj_nom +=1
            nom_lems.append(lemma.strip())
        if 'Case=Dat' in feats:
            nsubj_dat +=1
            dat_lems.append(lemma.strip())
        else:
            other +=1
nomlemma_nfreqs={l:[] for l in nom_lems}
for l in set(nom_lems):
    nomlemma_nfreqs[l] = nom_lems.count(l)/allrels
sorted1 = [(l, nomlemma_nfreqs[l]) for l in sorted(nomlemma_nfreqs, key=nomlemma_nfreqs.get, reverse=True)] #sorted list of tuples from the dic

datlemma_nfreqs={}
for l in set(dat_lems):
    datlemma_nfreqs[l] = dat_lems.count(l)/allrels
sorted2 = [(l, datlemma_nfreqs[l]) for l in sorted(datlemma_nfreqs, key=datlemma_nfreqs.get, reverse=True)]

print('Nominative sbjects in %s:\t%s\t%s' % (basename(input), nsubj_nom, nsubj_nom/allrels))
print(sorted1[:20]) #[value for (key, value) in sorted(numbers.items())]

print('==================================')

print('Dative sbjects in %s: %s' % (basename(input), nsubj_dat))
print(sorted2[:20])
print('Total nsubj: ', allsubj)
print(other)
print(allrels)




