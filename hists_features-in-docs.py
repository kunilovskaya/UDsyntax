#!/bin/python
# coding: utf-8
# slices a DataFrame and prints three hists to one axes
import sys,codecs,os
import pandas as pd
import numpy as np
from numpy import median
import matplotlib.pyplot as plt
import seaborn as sns

longt=pd.read_csv('/home/masha/aaa/oslo/bydoc_stats/m_bigtable.tsv', sep='\t') # reads into a DataFrame: rows of index and values
learners = longt[longt['class']=='learners']
prof = longt[longt['class']=='prof']
rnc = longt[longt['class']=='rnc']

col_labels = longt.columns
print(col_labels)

for feat in col_labels[2:]:
    #print(learners[feat])
    ax = sns.distplot(learners[feat], kde=False, bins=10, rug=False, hist_kws={"color": "black","histtype": "step", "linewidth": 2, "label":"learners", "alpha": 1})
    ax = sns.distplot(prof[feat], kde=False, bins=10, rug=False, hist_kws={"color": "grey","histtype": "step", "linewidth": 2,  "alpha": 1,"label":"prof"})
    ax = sns.distplot(rnc[feat], kde=False, bins=10, rug=False, hist_kws={"color": "red","histtype": "step", "linewidth": 2,  "alpha": 1, "label":"rnc"})

    ax.set(xlabel="probability of %s" %(feat), ylabel='how often this probability occurs in the data')
    plt.legend()
    plt.tight_layout()
    plt.savefig('/home/masha/aaa/oslo/plots/%s.png' %(feat), format='png')
    plt.show()
