#################################################################
##                       Load Libraries!                       ##
#################################################################
import numpy as np
import pandas as pd
import datetime
import time
import warnings

from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils import all_estimators
from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import *
import matplotlib.pyplot as plt
import xgboost
# import catboost
# import lightgbm
import plotly.graph_objs as go
import plotly.io as pio
import colorlover as cl
# Set default renderer for plotly
pio.renderers.default = 'notebook_connected'
# Configure warnings and pandas display options
warnings.filterwarnings("ignore")
pd.set_option("display.precision", 2)
pd.set_option("display.float_format", lambda x: "%.2f" % x)
from sklearn.model_selection import *
from IPython.display import display


##################################################################
##              Remove or Adding Class Classifiers              ##
##################################################################

removed_classifiers = [
    "ClassifierChain",
    "ComplementNB",
    "GradientBoostingClassifier",
    "GaussianProcessClassifier",
    "HistGradientBoostingClassifier",
    "MLPClassifier",
    "LogisticRegressionCV", 
    "MultiOutputClassifier", 
    "MultinomialNB", 
    "OneVsOneClassifier",
    "OneVsRestClassifier",
    "OutputCodeClassifier",
    "RadiusNeighborsClassifier",
    # "VotingClassifier",
]

CLASSIFIERS = [
    est
    for est in all_estimators()
    if (issubclass(est[1], ClassifierMixin) and (est[0] not in removed_classifiers))
]

CLASSIFIERS.append(("XGBClassifier", xgboost.XGBClassifier))
# CLASSIFIERS.append(("LGBMClassifier", lightgbm.LGBMClassifier))
# CLASSIFIERS.append(('CatBoostClassifier',catboost.CatBoostClassifier))

numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
)

categorical_transformer_low = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoding", OneHotEncoder(handle_unknown="ignore", sparse=False)),
    ]
)

categorical_transformer_high = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        # 'OrdianlEncoder' Raise a ValueError when encounters an unknown value. Check https://github.com/scikit-learn/scikit-learn/pull/13423
        ("encoding", OrdinalEncoder()),
    ]
)


# Helper function


def get_card_split(df, cols, n=11):
    """
    Splits categorical columns into 2 lists based on cardinality (i.e # of unique values)
    Parameters
    ----------
    df : Pandas DataFrame
        DataFrame from which the cardinality of the columns is calculated.
    cols : list-like
        Categorical columns to list
    n : int, optional (default=11)
        The value of 'n' will be used to split columns.
    Returns
    -------
    card_low : list-like
        Columns with cardinality < n
    card_high : list-like
        Columns with cardinality >= n
    """
    cond = df[cols].nunique() > n
    card_high = cols[cond]
    card_low = cols[~cond]
    return card_low, card_high



class rapidclassifier:

    def __init__(
        self,
        verbose=0,
        ignore_warnings=True,
        custom_metric=None,
        predictions=False,
        random_state=42,
        classifiers="all",
        
    ):
        self.verbose = verbose
        self.ignore_warnings = ignore_warnings
        self.custom_metric = custom_metric
        self.predictions = predictions
        self.models = {}
        self.random_state = random_state
        self.classifiers = classifiers

    def fit(self, X_train, X_test, y_train, y_test):
        """Fit Classification algorithms to X_train and y_train, predict and score on X_test, y_test.
        Parameters
        ----------
        X_train : array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.
        X_test : array-like,
            Testing vectors, where rows is the number of samples
            and columns is the number of features.
        y_train : array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.
        y_test : array-like,
            Testing vectors, where rows is the number of samples
            and columns is the number of features.
        Returns
        -------
        scores : Pandas DataFrame
            Returns metrics of all the models in a Pandas DataFrame.
        predictions : Pandas DataFrame
            Returns predictions of all the models in a Pandas DataFrame.
        """


        ###########################################################################
        ###########################################################################
        ###                                                                     ###
        ###                              METRICS!                               ###
        ###                                                                     ###
        ###########################################################################
        ###########################################################################

        Accuracy = []
        B_Accuracy = []
        ROC_AUC = []
        F1 = []
        names = []
        Recall = []
        Precision = []
        KFold_F1 = []
        TIME = []
        results = []
        predictions = {}

        ################################################################

        if self.custom_metric is not None:
            CUSTOM_METRIC = []

        if isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(X_train)
            X_test = pd.DataFrame(X_test)

        numeric_features = X_train.select_dtypes(include=[np.number]).columns
        categorical_features = X_train.select_dtypes(include=["object"]).columns

        categorical_low, categorical_high = get_card_split(
            X_train, categorical_features
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", numeric_transformer, numeric_features),
                ("categorical_low", categorical_transformer_low, categorical_low),
                ("categorical_high", categorical_transformer_high, categorical_high),
            ]
        )

        
        ###########################################################################

        if self.classifiers == "all":
            self.classifiers = CLASSIFIERS
        else:
            try:
                temp_list = []
                for classifier in self.classifiers:
                    full_name = (classifier.__name__, classifier)
                    temp_list.append(full_name)
                self.classifiers = temp_list
            except Exception as exception:
                print(exception)
                print("Invalid Classifier(s)")
        ###########################################################################

        
        for name, model in tqdm(self.classifiers):
            start = time.time()
            try:
                if "random_state" in model().get_params().keys():
                    pipe = Pipeline(
                        steps=[
                            ("preprocessor", preprocessor),
                            ("classifier", model(random_state=self.random_state)),
                        ]
                    )
                else:
                    pipe = Pipeline(
                        steps=[("preprocessor", preprocessor), ("classifier", model())]
                    )

                pipe.fit(X_train, y_train)
                # print(X_train)

                self.models[name] = pipe
                y_pred = pipe.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred, normalize=True)
                b_accuracy = balanced_accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="weighted")
                recall = recall_score(y_test, y_pred, average="weighted")
                precision = precision_score(y_test, y_pred, average="weighted")
                # print("precision : " ,precision)
                KF_5 = KFold(n_splits=5,  shuffle=True)
                # print("model" , model)
                f1_score_Kfold = cross_val_score(pipe, X=X_train, y=y_train, cv=KF_5,  scoring = "f1" ,   n_jobs=-1)
                ###########################################################################

                ###########################################################################

                ###########################################################################


                results.append(f1_score_Kfold)
                # print("results" , results)
                # names.append(name)
                # print(pipe)

                ###########################################################################
                ###########################################################################

                ###########################################################################

                f1_score_Kfold_mean = np.mean(f1_score_Kfold)
                # print("f1_score_Kfold_mean: " , f1_score_Kfold_mean)

                # print(f1_score_Kfold_mean)

                try:
                    roc_auc = roc_auc_score(y_test, y_pred)
                except Exception as exception:
                    roc_auc = None
                    if self.ignore_warnings is False:
                        print("ROC AUC couldn't be calculated for " + name)
                        print(exception)

                ###########################################################################
                names.append(name)
                ###########################################################################
                Accuracy.append(accuracy)
                ###########################################################################
                B_Accuracy.append(b_accuracy)
                ###########################################################################
                ROC_AUC.append(roc_auc)
                ###########################################################################
                F1.append(f1)
                ###########################################################################
                Recall.append(recall)
                ###########################################################################
                Precision.append(precision)
                ###########################################################################
                KFold_F1.append(f1_score_Kfold_mean)
                ###########################################################################
                TIME.append(time.time() - start)
                ###########################################################################

                if self.custom_metric is not None:
                    custom_metric = self.custom_metric(y_test, y_pred)
                    CUSTOM_METRIC.append(custom_metric)
                if self.verbose > 0:
                    if self.custom_metric is not None:
                        print("\n" ,
                            {
                                "Model": name,
                                "Accuracy": accuracy,
                                "Balanced Accuracy": b_accuracy,
                                "ROC AUC": roc_auc,    
                              "Recall" : recall_score,
                             "Precision" : precision,
                                "F1 Score": f1,
                             "5 Fold F1":f1_score_Kfold_mean ,
                                self.custom_metric.__name__: custom_metric,
                                "Time taken": time.time() - start,
                            }
                        )
                    else:
                        print( "\n" ,
                            {
                                "Model": name,
                                "Accuracy": accuracy,
                                "Balanced Accuracy": b_accuracy,
                                "ROC AUC": roc_auc, 
                             "Recall" : recall,     
                             "Precision" : precision,
                                "F1 Score": f1, 
                              "5 Fold F1": f1_score_Kfold_mean ,
                                "Time taken": time.time() - start,
                            }
                        )
                if self.predictions:
                    predictions[name] = y_pred
            except Exception as exception:
                if self.ignore_warnings is False:
                    print(name + " model failed to execute")
                    print(exception)

       
        if self.custom_metric is None:
            scores = pd.DataFrame(
                {
                    "Model": names,
                    "Accuracy": Accuracy,
                    "Balanced Accuracy": B_Accuracy,
                    "ROC AUC": ROC_AUC, 
                    "Recall" : Recall,     
                 "Precision" : Precision,
                    "F1 Score": F1,
                    "5 Fold F1": KFold_F1 ,
                    "Time Taken": TIME,
                }
            )
        else:
            scores = pd.DataFrame(
                {
                    "Model": names,
                    "Accuracy": Accuracy,
                    "Balanced Accuracy": B_Accuracy,
                    "ROC AUC": ROC_AUC,   
                  "Recall" : Recall, 
                  "Precision" : Precision,
                    "F1 Score": F1,
                  "5 Fold F1": KFold_F1 ,
                self.custom_metric.__name__: CUSTOM_METRIC,
                    "Time Taken": TIME,
                }
            )
        scores = scores.sort_values(by="F1 Score", ascending=False).set_index(
            "Model"
        )
                ###########################################################################

        if self.predictions:
            predictions_df = pd.DataFrame.from_dict(predictions)
        print("scores \n")
        display(scores)



        # compare_model_(results , names)
        return scores, predictions_df  if self.predictions is True else scores
    
    def generate_colors(self, n):
        colors = cl.scales[str(min(n, 12))]['qual']['Paired']
        while len(colors) < n:
            colors += colors
        return colors[:n]


        fig = go.Figure(data=data, layout=layout)
        fig.show()

    def provide_models(self, X_train, X_test, y_train, y_test):

        if len(self.models.keys()) == 0:
            self.fit(X_train, X_test, y_train, y_test) 

        return self.models