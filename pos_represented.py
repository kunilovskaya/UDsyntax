#!/usr/bin/python3
# pulls out the content of the 3d column (of conllu formatted files, 0-based) and puts them back together into a PoS represented texts
# keeps one-sent per line format and filters out sents >= 3 words; filters out PUNCT
# input: a folder of *.conllu; outputs pos-represented texts to a pos folder
# working dir = the input folder

import codecs, sys, os

arg = sys.argv[1]
folder = len(os.path.abspath(arg).split('/')) - 1
try:
    os.mkdir(arg + 'pos_'+ os.path.abspath(arg).split('/')[folder])
except FileExistsError:
    pass

fh = [f for f in os.listdir(arg) if f.endswith('.conllu')]
count_wds = 0
count_sents = 0
for f in fh:
    sents = []
    this_sent = []
    lines = codecs.open(arg + f, 'r', 'utf-8').readlines()
    out = codecs.open(arg + 'pos_' + os.path.abspath(arg).split('/')[folder]+ '/' + f + '.pos', 'w', 'utf-8')
    for line in lines:
        if line.strip() == '':
            if this_sent:
                sents.append(this_sent)
            this_sent = []
            continue
        if line.strip().startswith('#'):
            continue

        bits = line.strip().split('\t')
        try:
            if bits[3] != 'PUNCT':
                pos = bits[3]
                this_sent.append(pos)
        except IndexError:
            print(bits[0])

    if this_sent:
        sents.append(this_sent)

    sents2 = [sent for sent in sents if len(sent) >= 3]

    for sent in sents2:
        count_wds += len(sent)
        out.write(' '.join(sent))
        out.write('\n')
        print(sent, end = '\n', file=sys.stderr)
    out.close()
    count_sents += len(sents2)
print('Size(in tokens): ', count_wds, file=sys.stderr)
print('No. of sents : ', count_sents, file=sys.stderr)

print("Done. See the texts represented as pos-tags", file=sys.stderr)
