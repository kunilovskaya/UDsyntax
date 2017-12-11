#!/usr/bin/python3

import sys

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline


def visual(data, classes):
    # Here goes the 2-D plotting of the data...
    pca = PCA(n_components=2)
    x_r = pca.fit_transform(data)
    plt.figure()
    colors = ['darkorange', 'navy']
    if len(classes) == 3:
        colors = ['darkorange', 'navy', 'turquoise', ]
        # classes = ['rnc', 'learners', 'prof']
    lw = 2
    for color, target_name in zip(colors, classes):
        plt.scatter(x_r[group == target_name, 0], x_r[group == target_name, 1], s=10, color=color,
                    label=target_name, alpha=.8, lw=lw)
    plt.legend(loc='best', scatterpoints=1, prop={'size': 30})
    plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
    plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
    plt.show()
    plt.close()
    return x_r
    # Plotting finished.


if __name__ == "__main__":
    datafile = sys.argv[1]  # The path to the Big Table file

    train = pd.read_csv(datafile, header=0, delimiter="\t")

    # State the classification mode in the first argument!
    modes = ['3class', 'natVStran', 'natVSlearn', 'natVSprof', 'profVSlearn']

    mode = int(sys.argv[2])
    try:
        mode = modes[mode]
    except IndexError:
        print('Incorrect mode!', file=sys.stderr)
        exit()

    # State how many best features you want to use in the 2nd argument! (max=46)
    features = int(sys.argv[3])

    if features < 1 or features > len(train.columns) - 2:
        print('Incorrect number of features!', file=sys.stderr)
        exit()

    # Dataset preparation, depending on the chosen mode
    if mode == 'natVSlearn':
        train = train[train.group != 'prof']
    elif mode == 'natVSprof':
        train = train[train.group != 'learners']
    elif mode == 'profVSlearn':
        train = train[train.group != 'rnc']
    elif mode == 'natVStran':
        train.loc[train.group == 'prof', 'group'] = 'transl'
        train.loc[train.group == 'learners', 'group'] = 'transl'

    group = train['group']

    X = train[train.keys()[2:]]

    # Feature selection
    ff = SelectKBest(k=features).fit(X, group)
    X = SelectKBest(k=features).fit_transform(X, group)
    top_ranked_features = sorted(enumerate(ff.scores_), key=lambda x: x[1], reverse=True)[:features]
    top_ranked_features_indices = [x[0] for x in top_ranked_features]
    used_features = [train.keys()[x + 2] for x in top_ranked_features_indices]

    # Optionally print dataset characteristics:
    print(datafile, file=sys.stderr)
    print('Our mode:', mode, file=sys.stderr)
    print('Train data:', file=sys.stderr)
    print('Instances:', X.shape[0], file=sys.stderr)
    print('Features:', X.shape[1], file=sys.stderr)
    print('We use these best features (ranked by their importance):', used_features, file=sys.stderr)

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
    classifier = algo.fit(scaled_X, group)
    predicted = classifier.predict(scaled_X)

    print("Accuracy on the training set:", round(accuracy_score(train["group"], predicted), 3), file=sys.stderr)

    print(classification_report(train["group"], predicted), file=sys.stderr)

    print('Confusion matrix on the training set:', file=sys.stderr)
    print(confusion_matrix(train["group"], predicted), file=sys.stderr)

    print('=====', file=sys.stderr)
    print('Here goes cross-validation. Please wait a bit...', file=sys.stderr)

    averaging = True  # Do you want to average the cross-validate metrics?

    scoring = ['precision_macro', 'recall_macro', 'f1_macro']
    cv_scores = cross_validate(clf, X, group, cv=10, scoring=scoring, n_jobs=2)

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

    visual(scaled_X, classifier.classes_)
