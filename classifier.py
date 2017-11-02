#!/usr/bin/python3

import sys
import pandas as pd
from sklearn import preprocessing
from sklearn import svm
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn import tree
import graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

datafile = 'm_bigtable.tsv.gz'  # The path to the Big Table file

mode = int(sys.argv[1])

features = int(sys.argv[2])# How many best features you want to use? (max=46)

train = pd.read_csv(datafile, header=0, delimiter="\t")
group = train['class']

X = train[train.keys()[2:]]

ff = SelectKBest(k=features).fit(X, group)
X = SelectKBest(k=features).fit_transform(X, group)
top_ranked_features = sorted(enumerate(ff.scores_), key=lambda x: x[1], reverse=True)[:features]
top_ranked_features_indices = [x[0] for x in top_ranked_features]
used_features = [train.keys()[x+2] for x in top_ranked_features_indices]
print('We use these best features (ranked by their importance):', used_features, file=sys.stderr)

# Optionally print feature table size
print(datafile, 'Train data:', X.shape, file=sys.stderr)

# Optionally scaling the features
X = preprocessing.scale(X)

# Choosing the classifier:

#algo = DummyClassifier()
#algo = LogisticRegression(n_jobs=4, class_weight='balanced', max_iter=1000, solver='saga', multi_class="multinomial")
# algo = DecisionTreeClassifier(class_weight="balanced", max_depth=10)
algo = svm.SVC(class_weight="balanced")

# Uncomment this if you want to draw decision tree plot
#a = algo.fit(X, group)
#dot_data = StringIO()
#export_graphviz(a, out_file=dot_data,
#               feature_names=used_features,
#                class_names=a.classes_,
#               filled=True, rounded=True,
#                special_characters=True)
#pydotplus.graph_from_dot_data(dot_data.getvalue()).write_png(str(len(used_features))+".png")

clf = make_pipeline(preprocessing.StandardScaler(), algo)

classifier = clf.fit(X, group)
predicted = classifier.predict(X)

print("Accuracy on the training set:", round(accuracy_score(train["class"], predicted), 3), file=sys.stderr)

print(classification_report(train["class"], predicted), file=sys.stderr)

print('Confusion matrix on the training set:', file=sys.stderr)
print(confusion_matrix(train["class"], predicted), file=sys.stderr)


print('=====', file=sys.stderr)
print('Here goes cross-validation', file=sys.stderr)

scoring = ['precision_macro', 'recall_macro', 'f1_macro']
cv_scores = cross_validate(clf, X, group, cv=10, scoring=scoring)
print("Average Precision on 10-fold cross-validation: %0.3f (+/- %0.3f)" % (
    cv_scores['test_precision_macro'].mean(), cv_scores['test_precision_macro'].std() * 2), file=sys.stderr)
print("Average Recall on 10-fold cross-validation: %0.3f (+/- %0.3f)" % (
    cv_scores['test_recall_macro'].mean(), cv_scores['test_recall_macro'].std() * 2), file=sys.stderr)
print("Average F1 on 10-fold cross-validation: %0.3f (+/- %0.3f)" % (
    cv_scores['test_f1_macro'].mean(), cv_scores['test_f1_macro'].std() * 2), file=sys.stderr)
