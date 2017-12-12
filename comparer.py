#!/usr/bin/python3
import sys
from scipy import stats
import pandas as pd
from os.path import basename

# *.stats.tsv files
treebank_stats0 = sys.argv[1]
treebank_stats1 = sys.argv[2]

data0 = pd.read_csv(treebank_stats0, sep='\t')  # reads into a DataFrame: rows of index and values
data1 = pd.read_csv(treebank_stats1, sep='\t')

print('Feature\t%s\t%s\tt-value\tp-value' % (basename(treebank_stats0), basename(treebank_stats1)))
# .iterrows(): Iterate over DataFrame rows as (index, Series) pairs;
# zip iterates over rows of two dataframes in parallel?
for row0, row1 in zip(data0.iterrows(), data1.iterrows()):
    # get the content of the first cell in column (Feature, etc) from the first DataFrame (data0)
    # row0[1] ignores the tech info for each row in a DataFrame
    feature = row0[1]['Feature']
    average0 = row0[1]['Average']
    average1 = row1[1]['Average']
    std0 = row0[1]['Deviation']
    std1 = row1[1]['Deviation']
    len0 = row0[1]['Observations']
    len1 = row1[1]['Observations']
    if average0 == 0 or average1 == 0:
        ttest = ('nan', 'nan')
    else:
        ttest = stats.ttest_ind_from_stats(average1, std1, len1, average0, std0, len0, equal_var=False)
    print(feature + '\t', average0, '\t', average1, '\t', ttest[0], '\t', ttest[1])
