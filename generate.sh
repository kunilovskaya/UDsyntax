#!/bin/bash

cat header.txt >> ${1}.tsv

for i in ${1}/*.conllu
do
echo ${i}
./analyze_treebank.py ${i} >> ${1}.tsv
done