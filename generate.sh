#!/bin/bash

cat header.txt >> ${1}.tsv

# ${1} - это каталог, в котором находятся обрабатываемые файлы с расширениями conllu
# так же будет называться сгенерированный файл с таблицей (с расширением tsv).

for i in ${1}/*.conllu
do
echo ${i}
./analyze_treebank.py ${i} >> ${1}.tsv
done