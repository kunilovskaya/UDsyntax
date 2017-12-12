#!/usr/bin/python3

import os
import sys
import pandas as pd
from sklearn import preprocessing
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from classifier import visual, featureselection, selectmode


def preparedata(directory):
    ourdic = []
    print('Collecting data from the files...')
    for subdir in os.listdir(directory):
        files = [f for f in os.listdir(os.path.join(directory, subdir)) if f.endswith('.txt')]
        for f in files:
            rowdic = {'doc': f.replace('_pos.txt', ''), 'group': subdir.replace('pos_', '')}
            doc = open(os.path.join(directory, subdir, f))
            text = doc.read().strip().replace('\n', ' ')
            doc.close()
            rowdic['text'] = text
            ourdic.append(rowdic)
    ourdic = pd.DataFrame(ourdic)
    return ourdic


if __name__ == "__main__":
    datadir = 'pos'  # The path to where subdirectories whith text files are
    data = preparedata(datadir)

    ourmode = int(sys.argv[1])

    data = selectmode(data, ourmode)

    groups = data['group']

    features = None
    # Optionally state how many best features you want to use in the 2nd argument.
    if len(sys.argv) > 2:
        features = int(sys.argv[2])

    vectorizer = CountVectorizer(lowercase=False, ngram_range=(3, 3))  # State min and max N-grams.
    train_data_features = vectorizer.fit_transform(data['text'])
    train_data_features = train_data_features.todense()
    vocabulary = vectorizer.get_feature_names()

    if features:
        train_data_features, top_feat = featureselection(train_data_features, groups, features)
    else:
        train_data_features, top_feat = featureselection(train_data_features, groups, len(vocabulary))

    # Optionally print dataset characteristics:
    print('Our mode:', ourmode)
    print('Train data:')
    print('Instances:', train_data_features.shape[0])
    print('Features:', train_data_features.shape[1])
    print('We use these best features (ranked by their importance):', [vocabulary[x] for x in top_feat][:100], '...')

    # Optionally scaling the features
    scaled_X = preprocessing.scale(train_data_features)

    print("The data is ready! Let's train some models...")

    algo = svm.SVC(class_weight="balanced")

    clf = make_pipeline(preprocessing.StandardScaler(), algo)

    classifier = algo.fit(scaled_X, groups)
    predicted = classifier.predict(scaled_X)

    print("Accuracy on the training set:", round(accuracy_score(data["group"], predicted), 3))

    print(classification_report(data["group"], predicted))

    print('Confusion matrix on the training set:')
    print(confusion_matrix(data["group"], predicted))

    # visual(scaled_X, groups, classifier.classes_)

    print('=====')
    print('Here goes cross-validation. Please wait a bit...')

    averaging = True  # Do you want to average the cross-validate metrics?

    scoring = ['precision_macro', 'recall_macro', 'f1_macro']
    cv_scores = cross_validate(clf, train_data_features, groups, cv=10, scoring=scoring, n_jobs=2)

    if averaging:
        print("Average Precision on 10-fold cross-validation: %0.3f (+/- %0.3f)" % (
            cv_scores['test_precision_macro'].mean(), cv_scores['test_precision_macro'].std() * 2))
        print("Average Recall on 10-fold cross-validation: %0.3f (+/- %0.3f)" % (
            cv_scores['test_recall_macro'].mean(), cv_scores['test_recall_macro'].std() * 2))
        print("Average F1 on 10-fold cross-validation: %0.3f (+/- %0.3f)" % (
            cv_scores['test_f1_macro'].mean(), cv_scores['test_f1_macro'].std() * 2))
    else:
        print("Precision values on 10-fold cross-validation:")
        print(cv_scores['test_precision_macro'])
        print("Recall values on 10-fold cross-validation:")
        print(cv_scores['test_recall_macro'])
        print("F1 values on 10-fold cross-validation:")
        print(cv_scores['test_f1_macro'])
