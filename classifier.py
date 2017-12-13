#!/usr/bin/python3

import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest


def selectmode(x, umode):
    # State the classification mode in the first argument!
    modes = ['3class', 'natVStran', 'natVSlearn', 'natVSprof', 'profVSlearn', 'enprofVSenlearn']
    try:
        umode = modes[umode]
    except IndexError:
        print('Incorrect mode!', file=sys.stderr)
        exit()
    # Dataset preparation, depending on the chosen mode
    newdata = x
    exclusion = []
    if umode == 'natVSlearn':
        exclusion = ['prof', 'en_learners', 'en_prof']
    elif umode == 'natVSprof':
        exclusion = ['learner', 'en_learners', 'en_prof']
    elif umode == 'profVSlearn':
        exclusion = ['rnc', 'en_learners', 'en_prof']
    elif umode == 'enprofVSenlearn':
        exclusion = ['rnc', 'learner', 'prof']
    elif umode == '3class':
        exclusion = ['en_learners', 'en_prof']
    elif umode == 'natVStran':
        exclusion = ['en_learners', 'en_prof']
        newdata.loc[newdata.group == 'prof', 'group'] = 'transl'
        newdata.loc[newdata.group == 'learner', 'group'] = 'transl'
    newdata = newdata[~newdata.group.isin(exclusion)]
    return newdata


def featureselection(x, labels, n_features):
    # Feature selection
    ff = SelectKBest(k=n_features).fit(x, labels)
    newdata = SelectKBest(k=n_features).fit_transform(x, labels)
    top_ranked_features = sorted(enumerate(ff.scores_), key=lambda y: y[1], reverse=True)[:n_features]
    top_ranked_features_indices = [x[0] for x in top_ranked_features]
    return newdata, top_ranked_features_indices


def visual(data, labels, classes):
    # Here goes the 2-D plotting of the data...
    pca = PCA(n_components=2)
    x_r = pca.fit_transform(data)
    plt.figure()
    # consistent colors
    colors = {'learner': 'navy', 'prof': 'turquoise', 'rnc': 'darkorange', 'transl': 'blue', }
    lw = 2
    if 'transl' not in classes:
        classes = reversed(classes)
    for target_name in classes:
        plt.scatter(x_r[labels == target_name, 0], x_r[labels == target_name, 1], s=1, color=colors[target_name],
                    label=target_name, alpha=.8, lw=lw)
    plt.legend(loc='best', scatterpoints=1, prop={'size': 15})
    plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
    plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
    #plt.savefig('plot.png', dpi=300)
    plt.show()
    plt.close()
    return x_r
    # Plotting finished.


if __name__ == "__main__":
    datafile = sys.argv[1]  # The path to the Big Table file

    if datafile.endswith('.gz'):
        train = pd.read_csv(datafile, compression='gzip', header=0, delimiter="\t")
    else:
        train = pd.read_csv(datafile, header=0, delimiter="\t")

    # State the classification mode in the first argument!
    mode = int(sys.argv[2])

    features = None
    # Optionally state how many best features you want to use in the 2nd argument.
    if len(sys.argv) > 3:
        features = int(sys.argv[3])

    train = selectmode(train, mode)

    groups = train['group']

    training_set = train.drop(columns=['doc', 'group'])

    if features:
        if features < 1 or features > len(training_set.columns):
            print('Incorrect number of features!', file=sys.stderr)
            exit()
        X, top_feat = featureselection(training_set, groups, features)
    else:
        X, top_feat = featureselection(training_set, groups, len(training_set.columns))

    # Optionally print dataset characteristics:
    print(datafile, file=sys.stderr)
    print('Our mode:', mode, file=sys.stderr)
    print('Train data:', file=sys.stderr)
    print('Instances:', X.shape[0], file=sys.stderr)
    print('Features:', X.shape[1], file=sys.stderr)
    print('We use these best features (ranked by their importance):',
          [training_set.keys()[x] for x in top_feat], file=sys.stderr)

    # Optionally scaling the features
    scaled_X = preprocessing.scale(X)

    # Choosing the classifier:

    # algo = DummyClassifier()
    # algo = DecisionTreeClassifier(class_weight="balanced", max_depth=10)
    algo = svm.SVC(class_weight="balanced")

    # Uncomment this if you want to draw decision tree plot
    # from sklearn import tree
    # import graphviz
    # from sklearn.externals.six import StringIO
    # from sklearn.tree import export_graphviz
    # import pydotplus
    # a = algo.fit(X, group)
    # dot_data = StringIO()
    # export_graphviz(a, out_file=dot_data,
    #               feature_names=used_features,
    #                class_names=a.classes_,
    #               filled=True, rounded=True,
    #                special_characters=True)
    # pydotplus.graph_from_dot_data(dot_data.getvalue()).write_png(str(len(used_features))+".png")

    print("The data is ready! Let's train some models...", file=sys.stderr)
    clf = make_pipeline(preprocessing.StandardScaler(), algo)
    classifier = algo.fit(scaled_X, groups)
    predicted = classifier.predict(scaled_X)

    print("Accuracy on the training set:", round(accuracy_score(groups, predicted), 3), file=sys.stderr)

    print(classification_report(groups, predicted), file=sys.stderr)

    print('Confusion matrix on the training set:', file=sys.stderr)

    print('=====', file=sys.stderr)
    print('Here goes cross-validation. Please wait a bit...', file=sys.stderr)

    averaging = True  # Do you want to average the cross-validate metrics?

    scoring = ['precision_macro', 'recall_macro', 'f1_macro']
    cv_scores = cross_validate(clf, X, groups, cv=10, scoring=scoring, n_jobs=2)

    if averaging:
        print("Average Precision on 10-fold cross-validation: %0.3f (+/- %0.3f)" % (
            cv_scores['test_precision_macro'].mean(), cv_scores['test_precision_macro'].std() * 2), file=sys.stderr)
        print("Average Recall on 10-fold cross-validation: %0.3f (+/- %0.3f)" % (
            cv_scores['test_recall_macro'].mean(), cv_scores['test_recall_macro'].std() * 2), file=sys.stderr)
        print("Average F1 on 10-fold cross-validation: %0.3f (+/- %0.3f)" % (
            cv_scores['test_f1_macro'].mean(), cv_scores['test_f1_macro'].std() * 2), file=sys.stderr)
    else:
        print("Precision values on 10-fold cross-validation:", file=sys.stderr)
        print(cv_scores['test_precision_macro'], file=sys.stderr)
        print("Recall values on 10-fold cross-validation:", file=sys.stderr)
        print(cv_scores['test_recall_macro'], file=sys.stderr)
        print("F1 values on 10-fold cross-validation:", file=sys.stderr)
        print(cv_scores['test_f1_macro'], file=sys.stderr)

    visual(scaled_X, groups, classifier.classes_)



