# RapidPredict
RapidPredict is a Python library that simplifies the process of fitting and evaluating multiple machine learning models from scikit-learn. It's designed to provide a quick way to test various algorithms on a given dataset and compare their performance. 


# Installation

To install Rapid Predict from PyPI:

    pip install rapidpredict

# Usage

To use Rapid Predict in a project:

    import rapidpredict



## Classification

Example :

    from rapidpredict import classification as rp
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    data = load_breast_cancer()
    X = data.data
    y= data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.5,random_state =123)

    clf = rp(verbose=0,ignore_warnings=True, custom_metric=None)
    models,predictions = clf.fit(X_train, X_test, y_train, y_test)



    |Model |Accuracy	 |Balanced Accuracy |	ROC | AUC |	Recall |	Precision | F1 Score |	5 Fold F1 |	Time  Taken |
    |-------------------------------|------|------|------|------|------|------|------|------|
    | QuadraticDiscriminantAnalysis | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.96 | 0.09 |
    | RandomForestClassifier        | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.96 | 1.21 |
    | LogisticRegression            | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.98 | 0.17 |
    | ExtraTreesClassifier          | 0.99 | 0.98 | 0.98 | 0.99 | 0.99 | 0.99 | 0.97 | 0.80 |
    | RidgeClassifier               | 0.99 | 0.98 | 0.98 | 0.99 | 0.99 | 0.99 | 0.96 | 0.13 |
    | LinearSVC                     | 0.99 | 0.98 | 0.98 | 0.99 | 0.99 | 0.99 | 0.96 | 0.10 |
    | SVC                           | 0.99 | 0.98 | 0.98 | 0.99 | 0.99 | 0.99 | 0.97 | 0.10 |
    | RidgeClassifierCV             | 0.99 | 0.98 | 0.98 | 0.99 | 0.99 | 0.99 | 0.96 | 0.17 |
    | LabelPropagation              | 0.98 | 0.98 | 0.98 | 0.98 | 0.98 | 0.98 | 0.94 | 0.17 |
    | LabelSpreading                | 0.98 | 0.98 | 0.98 | 0.98 | 0.98 | 0.98 | 0.96 | 0.19 |
    | SGDClassifier                 | 0.98 | 0.98 | 0.98 | 0.98 | 0.98 | 0.98 | 0.97 | 0.09 |
    | Perceptron                    | 0.98 | 0.98 | 0.98 | 0.98 | 0.98 | 0.98 | 0.97 | 0.08 |
    | KNeighborsClassifier          | 0.98 | 0.98 | 0.98 | 0.98 | 0.98 | 0.98 | 0.97 | 0.11 |
    | DecisionTreeClassifier        | 0.98 | 0.98 | 0.98 | 0.98 | 0.98 | 0.98 | 0.93 | 0.09 |
    | BernoulliNB                   | 0.98 | 0.98 | 0.98 | 0.98 | 0.98 | 0.98 | 0.93 | 0.09 |
    | LinearDiscriminantAnalysis    | 0.98 | 0.97 | 0.97 | 0.98 | 0.98 | 0.98 | 0.96 | 0.14 |
    | CalibratedClassifierCV        | 0.98 | 0.97 | 0.97 | 0.98 | 0.98 | 0.98 | 0.97 | 0.24 |
    | AdaBoostClassifier            | 0.97 | 0.97 | 0.97 | 0.97 | 0.97 | 0.97 | 0.95 | 0.89 |
    | PassiveAggressiveClassifier   | 0.97 | 0.97 | 0.97 | 0.97 | 0.97 | 0.97 | 0.97 | 0.09 |
    | XGBClassifier                 | 0.97 | 0.97 | 0.97 | 0.97 | 0.97 | 0.97 | 0.97 | 0.45 |
    | BaggingClassifier             | 0.97 | 0.96 | 0.96 | 0.97 | 0.97 | 0.97 | 0.95 | 0.32 |
    | NuSVC                         | 0.97 | 0.95 | 0.95 | 0.97 | 0.97 | 0.96 | 0.95 | 0.12 |
    | NearestCentroid               | 0.97 | 0.95 | 0.95 | 0.97 | 0.97 | 0.96 | 0.94 | 0.08 |
    | GaussianNB                    | 0.97 | 0.95 | 0.95 | 0.97 | 0.97 | 0.96 | 0.94 | 0.08 |
    | ExtraTreeClassifier           | 0.94 | 0.94 | 0.94 | 0.94 | 0.94 | 0.94 | 0.93 | 0.08 |
    | DummyClassifier               | 0.62 | 0.50 | 0.50 | 0.62 | 0.39 | 0.48 | 0.77 | 0.08 |



This code updated from github  ["Lazypredic-Shankar Rao Pandala"](https://github.com/shankarpandala/lazypredict) 
