#################################################################
##                       Load Libraries!                       ##
#################################################################
import numpy as np
import pandas as pd
import datetime
import time
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

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

##################################################################
##                    Helper function                           ##
##################################################################

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


##################################################################
##                    Class rapidclassifier                     ##
##################################################################

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
    
    def provide_models(self, X_train, X_test, y_train, y_test):

        if len(self.models.keys()) == 0:
            self.fit(X_train, X_test, y_train, y_test) 

        return self.models
    

##################################################################
##                    Class plot_target                         ##
##################################################################

class plot_target:
    def __init__(self, dataset, target="target"):
        self.dataset = dataset
        self.target = target
        self.plot()

    def plot(self):
        # Create a DataFrame containing the target column
        target_df = pd.DataFrame({self.target: self.dataset})
        counts = target_df[self.target].value_counts()

        # Create a subplot with 1 row and 2 columns
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))

        # Create the countplot in the first subplot
        sns.countplot(x=self.target, data=target_df, ax=ax[0])
        ax[0].set_title("Target Distribution (Countplot)")
        ax[0].set_xlabel("Target")
        ax[0].set_ylabel("Count")

        # Create the pie chart in the second subplot
        ax[1].pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
        ax[1].set_title("Target Distribution (Pie Chart)")
        ax[1].axis('equal')  # Equal aspect ratio ensures the pie chart is circular

        plt.show()

    def __repr__(self):
        return ''
    




##################################################################
##                  class compareModels_bargraph                ##                    
##################################################################


class compareModels_bargraph:
    def __init__(self, model_names, model_metrics_flscore):
        self.model_names = model_names.to_list()
        self.model_metrics_flscore = model_metrics_flscore.to_list()
        self.df = pd.DataFrame({'Model': self.model_names, 'F1 Score': self.model_metrics_flscore})
        self.plot()

    def plot(self):
        df_sorted = self.df.sort_values('F1 Score', ascending=True)

        fig, ax = plt.subplots(figsize=(16, 8))

        # Generate an array of 26 different colors using a colormap
        colormap = plt.cm.get_cmap("tab20", 26)
        colors = colormap(np.arange(26))

        # Pass the colors array to the barh function
        bars = ax.barh(df_sorted['F1 Score'], df_sorted['Model'], color=colors)

        ax.set_title('Model F1 Score')
        ax.set_xlabel('F1 Score')
        ax.set_ylabel('Model')

        ax.invert_yaxis()
             # Add model accuracies at the end of each bar
        for i, bar in enumerate(bars):
            width = bar.get_width()
            label = f'{width:.2f}'
            ax.text(width, bar.get_y() + bar.get_height() / 2, label, ha='left', va='center')


        plt.show()
        def __repr__(self):
        # Return an empty string or a custom message
            return ''






##################################################################
##                  class compareModels_boxplot                 ##                    
##################################################################

class compareModels_boxplot:
    def __init__(self, model_names, model_metrics_flscore):
        self.model_names = model_names.to_list()
        self.model_metrics_flscore = model_metrics_flscore.to_list()
        self.df = pd.DataFrame({'Model': self.model_names, 'F1 Score': self.model_metrics_flscore})
        self.plot()

    def plot(self):
        df_sorted = self.df.sort_values('F1 Score', ascending=True)

        fig, ax =plt.subplots(figsize=(10, 4)) 

        # Generate an array of 26 different colors using a colormap
        colormap = plt.cm.get_cmap("tab20", 26)
        colors = colormap(np.arange(26))

        # Create a custom boxplot with seaborn
        sns.boxplot(x='F1 Score', y='Model', data=df_sorted, palette=colors[:len(self.model_names)], ax=ax )

        ax.set_title('Model F1 Scores')
        ax.set_xlabel('F1 Score')
        ax.set_ylabel('Model')
        ax.xaxis.set_tick_params(rotation=90)

        #     # Add model accuracies close to the model names
        # for i, row in df_sorted.iterrows():
        #     ax.text(row['F1 Score'] + 0.005, i, f"{row['F1 Score']:.3f}", va='center')


        plt.show()
        def __repr__(self):
        # Return an empty string or a custom message
            return ''



# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# import seaborn as sns

# class CompareModel_boxplot:
#     def __init__(self, model_names, model_accuracies):
#         self.model_names = model_names.to_list()
#         self.model_accuracies = model_accuracies.to_list()
#         self.df = pd.DataFrame({'Model': self.model_names, 'F1 Score': self.model_accuracies})
#         self.plot()

#     def plot(self):
#         self.df['F1 Score'] = pd.to_numeric(self.df['F1 Score'])  # Convert F1 Score column to numeric data type

#         df_sorted = self.df.sort_values('F1 Score', ascending=True)
#         best_f1 = df_sorted['F1 Score'].max()
#         best_model_idx = df_sorted['F1 Score'].idxmax()
#         best_model = df_sorted.loc[best_model_idx]['Model']

#         fig, ax = plt.subplots(figsize=(12, 6))  # Increase the figure size

#         # Generate an array of 26 different colors using a colormap
#         colormap = plt.cm.get_cmap("tab20", 26)
#         colors = colormap(np.arange(26))

#         # Create a custom boxplot with seaborn
#         sns.boxplot(x='F1 Score', y='Model', data=df_sorted, palette=colors[:len(self.model_names)], ax=ax)

#         ax.set_title('Model Accuracies')
#         ax.set_xlabel('F1 Score')
#         ax.set_ylabel('Model')

#         # Rotate the xtick labels by 90 degrees
#         ax.set_xticklabels(ax.get_xticks(), rotation=90)

#         # Add accuracies to the plot
#         for i, row in df_sorted.iterrows():
#             ax.text(row['F1 Score'], i, f"{row['F1 Score']:.2f}", color='black', ha="left", va="center")

#         # Mention the best result
#         ax.text(0.98, 0.02, f"Best result: {best_model} ({best_f1:.2f})", transform=ax.transAxes, ha="right", va="bottom", fontsize=12)

#         plt.show()


