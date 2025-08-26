import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import explained_variance_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import ParameterGrid
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR


from tqdm import tqdm

CATEGORICAL = "categorical"
CONTINUOUS = "continuous"
ORDINAL = "ordinal"
INPUT_SIZE = None

_MODELS = None

def default_models():
    r = {
    'binary_classification': [ # 184
         # {
         #     'class': DecisionTreeClassifier, # 48
         #     'kwargs': {
         #         'max_depth': [4, 8, 16, 32],
         #         'min_samples_split': [2, 4, 8],
         #         'min_samples_leaf': [1, 2, 4, 8]
         #     }
         # },
         # {
         #     'class': AdaBoostClassifier, # 4
         #     'kwargs': {
         #         'n_estimators': [10, 50, 100, 200]
         #     }
         # },
         # {
         #    'class': LogisticRegression, # 36
         #    'kwargs': {
         #         'solver': ['lbfgs'],
         #         'n_jobs': [-1],
         #         'max_iter': [10, 50, 100, 200],
         #         'C': [0.01, 0.1, 1.0],
         #         'tol': [1e-01, 1e-02, 1e-04]
         #     }
         # },
        {
            'class': MLPClassifier, # 12
            'kwargs': {
                'hidden_layer_sizes': [(int((INPUT_SIZE+1)*2/3))],
                'solver': ['adam'],                 # 优化算法
                'learning_rate_init': [0.001],      # 学习率
                'max_iter': [200],                  # 最大迭代次数 (epochs)
                # 'max_iter': [50, 100],
                # 'alpha': [0.0001, 0.001],
                'activation':['relu']
            }
        },
        # {
        #     'class': RandomForestClassifier, # 48
        #     'kwargs': {
        #          'max_depth': [8, 16, None],
        #          'min_samples_split': [2, 4, 8],
        #          'min_samples_leaf': [1, 2, 4, 8],
        #         'n_jobs': [-1]
        #
        #     }
        # },
        #     {
        #     'class': KNeighborsClassifier, # 48
        #     'kwargs': {
        #         'n_neighbors': [3, 5, 7, 9, 11],
        #         'weights': ['uniform', 'distance'],
        #         'metric': ['euclidean', 'manhattan', 'minkowski']
        #
        #     }
        # },
        #{
        #    'class': SVR,  # 48
        #    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        #    'C': [0.1, 1, 10, 100],
        #    'gamma': ['scale', 'auto'] + [0.001, 0.01, 0.1, 1, 10],
        #    'degree': [2, 3, 4]  # 仅适用于多项式核
        #},

        # {
        #     'class': XGBClassifier, # 36
        #     'kwargs': {
        #          'n_estimators': [10, 50, 100],
        #          'min_child_weight': [1, 10],
        #          'max_depth': [5, 10, 20],
        #          'gamma': [0.0, 1.0],
        #          'objective': ['binary:logistic'],
        #          'nthread': [-1]
        #     },
        # }

    ],
    'multiclass_classification': [ # 132

        {
            'class': MLPClassifier, # 12
            'kwargs': {
                'hidden_layer_sizes': [(int((INPUT_SIZE+1)*2/3))],
                'solver': ['adam'],                 # 优化算法
                'learning_rate_init': [0.001],      # 学习率
                'max_iter': [200],                  # 最大迭代次数 (epochs)
                # 'max_iter': [50, 100],
                # 'alpha': [0.0001, 0.001],
                'activation':['relu']
        },
        }
        #  {
        #      'class': DecisionTreeClassifier, # 48
        #      'kwargs': {
        #          'max_depth': [4, 8, 16, 32],
        #          'min_samples_split': [2, 4, 8],
        #          'min_samples_leaf': [1, 2, 4, 8]
        #      }
        #  },
        # {
        #     'class': RandomForestClassifier, # 36
        #     'kwargs': {
        #          'max_depth': [8, 16, None],
        #          'min_samples_split': [2, 4, 8],
        #          'min_samples_leaf': [1, 2, 4, 8],
        #          'n_jobs': [-1]
        #
        #     }
        # },
        # {
        #     'class': XGBClassifier, # 36
        #     'kwargs': {
        #          'n_estimators': [10, 50, 100],
        #          'min_child_weight': [1, 10],
        #          'max_depth': [5, 10, 20],
        #          'gamma': [0.0, 1.0],
        #          'objective': ['binary:logistic'],
        #          'nthread': [-1]
        #     }
        # }

    ],
    'regression': [ # 84
        # {
        #     'class': LinearRegression,
        # },
        {
            'class': MLPRegressor, # 12
            'kwargs': {
                'hidden_layer_sizes': [(100, ), (200, ), (100, 100)],
                'max_iter': [50, 100],
                'alpha': [0.0001, 0.001]
            }
        },
        {
            'class': XGBRegressor, # 36
            'kwargs': {
                 'n_estimators': [10, 50, 100],
                 'min_child_weight': [1, 10],
                 'max_depth': [5, 10, 20],
                 'gamma': [0.0, 1.0],
                 'objective': ['reg:linear'],
                 'nthread': [-1]
            }
        },
        # {
        #     'class': RandomForestRegressor, # 36
        #     'kwargs': {
        #          'max_depth': [8, 16, None],
        #          'min_samples_split': [2, 4, 8],
        #          'min_samples_leaf': [1, 2, 4, 8],
        #          'n_jobs': [-1]
        #     }
        # }
    ]
}
    return r

class FeatureMaker:

    def __init__(self, metadata, label_column='label', label_type='int', sample=50000):
        self.columns = metadata['columns']
        self.label_column = label_column
        self.label_type = label_type
        self.sample = sample
        self.encoders = dict()

    def make_features(self, data):
        data = data.copy()
        np.random.shuffle(data)
        data = data[:self.sample]

        features = []
        labels = []

        for index, cinfo in enumerate(self.columns):
            col = data[:, index]
            if cinfo['name'] == self.label_column:
                if self.label_type == 'int':
                    # add a function: to map the label to a index
                    index_dic = list(set(col))
                    labels = np.array(list(map(index_dic.index, col)))
                    # labels = col.astype(int)
                elif self.label_type == 'float':
                    labels = col.astype(float)
                else:
                    assert 0, 'unkown label type'
                continue

            if cinfo['type'] == CONTINUOUS:
                cmin = cinfo['min']
                cmax = cinfo['max']
                if cmin >= 0 and cmax >= 1e3:
                    feature = np.log(np.maximum(col.astype('float'), 1e-2))

                elif cmin == cmax ==0: # when the o column
                    feature = col
                else:
                    feature = (col - cmin) / (cmax - cmin) * 5

            elif cinfo['type'] == ORDINAL:
                feature = col

            else:
                if cinfo['size'] <= 2:
                    feature = col

                else:
                    encoder = self.encoders.get(index)
                    col = col.reshape(-1, 1)
                    if encoder:
                        feature = encoder.transform(col)
                    else:
                        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                        self.encoders[index] = encoder
                        feature = encoder.fit_transform(col)

            features.append(feature)

        features = np.column_stack(features)

        return features, labels


def _prepare_ml_problem(train, val, test, metadata):  # fake train test
    fm = FeatureMaker(metadata)
    x_trains, y_trains = [], []

    for i in train:
        x_train, y_train = fm.make_features(i)
        x_trains.append(x_train)
        y_trains.append(y_train)

    x_val, y_val = fm.make_features(val)
    x_test, y_test = fm.make_features(test)
    model = _MODELS[metadata['problem_type']]

    return x_trains, y_trains, x_val, y_val, x_test, y_test, model


def _weighted_f1(y_test, pred):
    report = classification_report(y_test, pred, output_dict=True)
    classes = list(report.keys())[:-3]
    proportion = [  report[i]['support'] / len(y_test) for i in classes]
    try:
        weighted_f1 = np.sum(list(map(lambda i, prop: report[i]['f1-score']* (1-prop)/(len(classes)-1), classes, proportion)))
    except:
        weighted_f1 = -1
    return weighted_f1 


@ignore_warnings(category=ConvergenceWarning)
def _evaluate_multi_classification(train, test, fake, metadata):
    x_trains, y_trains, x_valid, y_valid, x_test, y_test, classifiers = _prepare_ml_problem(fake, train, test, metadata)
    best_f1_scores = []
    best_weighted_scores = []
    best_auroc_scores = []
    best_acc_scores = []
    best_avg_scores = []

    for model_spec in classifiers:
        model_class = model_spec['class']
        model_kwargs = model_spec.get('kwargs', dict())
        model_repr = model_class.__name__

        unique_labels = np.unique(y_trains[0])

        param_set = list(ParameterGrid(model_kwargs))

        results = []
        for param in tqdm(param_set):
            model = model_class(**param)

            try:
                model.fit(x_trains[0], y_trains[0])
            except:
                pass
            
            if len(unique_labels) != len(np.unique(y_valid)):
                pred = [unique_labels[0]] * len(x_valid)
                pred_prob = np.array([1.] * len(x_valid))
            else:
                pred = model.predict(x_valid)
                pred_prob = model.predict_proba(x_valid)

            macro_f1 = f1_score(y_valid, pred, average='macro')
            weighted_f1 = _weighted_f1(y_valid, pred)
            acc = accuracy_score(y_valid, pred)

            # 3. auroc
            size = [a["size"] for a in metadata["columns"] if a["name"] == "label"][0]
            rest_label = set(range(size)) - set(unique_labels)
            tmp = []
            j = 0
            for i in range(size):
                if i in rest_label:
                    tmp.append(np.array([0] * y_valid.shape[0])[:,np.newaxis])
                else:
                    try:
                        tmp.append(pred_prob[:,[j]])
                    except:
                        tmp.append(pred_prob[:, np.newaxis])
                    j += 1
            try:
                roc_auc = roc_auc_score(np.eye(size)[y_valid], np.hstack(tmp), multi_class='ovr')
            except:
                roc_auc = -1
            results.append(
                {   
                    "name": model_repr,
                    "param": param,
                    "macro_f1": macro_f1,
                    "weighted_f1": weighted_f1,
                    "roc_auc": roc_auc, 
                    "accuracy": acc
                }
            )

        results = pd.DataFrame(results)
        results['avg'] = results.loc[:, ['macro_f1', 'weighted_f1', 'roc_auc']].mean(axis=1)        
        best_f1_param = results.param[results.macro_f1.idxmax()]
        best_weighted_param = results.param[results.weighted_f1.idxmax()]
        best_auroc_param = results.param[results.roc_auc.idxmax()]
        best_acc_param = results.param[results.accuracy.idxmax()]
        best_avg_param = results.param[results.avg.idxmax()]


        # test the best model
        results = pd.DataFrame(results)
        # best_param = results.param[results.macro_f1.idxmax()]

        def _calc(best_model):
            best_scores = []
            for x_train, y_train in zip(x_trains, y_trains):
                try:
                    best_model.fit(x_train, y_train)
                except:
                    pass 
                
                if len(unique_labels) != len(np.unique(y_test)):
                    pred = [unique_labels[0]] * len(x_test)
                    pred_prob = np.array([1.] * len(x_test))
                else:
                    pred = best_model.predict(x_test)
                    pred_prob = best_model.predict_proba(x_test)

                macro_f1 = f1_score(y_test, pred, average='macro')
                precision = precision_score(y_test, pred, average='macro')
                recall = recall_score(y_test, pred, average='macro')
                weighted_f1 = _weighted_f1(y_test, pred)
                acc = accuracy_score(y_test, pred)

                # 3. auroc
                size = [a["size"] for a in metadata["columns"] if a["name"] == "label"][0]
                rest_label = set(range(size)) - set(unique_labels)
                tmp = []
                j = 0
                for i in range(size):
                    if i in rest_label:
                        tmp.append(np.array([0] * y_test.shape[0])[:,np.newaxis])
                    else:
                        try:
                            tmp.append(pred_prob[:,[j]])
                        except:
                            tmp.append(pred_prob[:, np.newaxis])
                        j += 1
                try:
                    roc_auc = roc_auc_score(np.eye(size)[y_test], np.hstack(tmp), multi_class='ovr')
                except:
                    roc_auc = -1
                best_scores.append(
                    {   
                        # "name": model_repr,
                        "macro_precision": precision,
                        "macro_recall":recall,
                        "macro_f1": macro_f1,
                        "weighted_f1": weighted_f1,
                        "roc_auc": roc_auc, 
                        "accuracy": acc
                    }
                )
            return pd.DataFrame(best_scores).mean(axis=0)

        def _df(dataframe):
            return {
                "name": model_repr,
                "accuracy": dataframe.accuracy,
                "macro_precision": dataframe.macro_precision,
                "macro_recall": dataframe.macro_recall,
                "macro_f1": dataframe.macro_f1,

            }

        best_f1_scores.append(_df(_calc(model_class(**best_f1_param))))
        best_weighted_scores.append(_df(_calc(model_class(**best_weighted_param))))
        best_auroc_scores.append(_df(_calc(model_class(**best_auroc_param))))
        best_acc_scores.append(_df(_calc(model_class(**best_acc_param))))
        best_avg_scores.append(_df(_calc(model_class(**best_avg_param))))

    return pd.DataFrame(best_f1_scores), pd.DataFrame(best_weighted_scores), pd.DataFrame(best_auroc_scores)


@ignore_warnings(category=ConvergenceWarning)
def _evaluate_binary_classification(train, test, fake, metadata):
    x_trains, y_trains, x_valid, y_valid, x_test, y_test, classifiers = _prepare_ml_problem(fake, train, test, metadata)

    best_f1_scores = []
    best_weighted_scores = []
    best_auroc_scores = []
    best_acc_scores = []
    best_avg_scores = []

    for model_spec in classifiers:
        model_class = model_spec['class']
        model_kwargs = model_spec.get('kwargs', dict())
        model_repr = model_class.__name__

        unique_labels = np.unique(y_trains[0])

        param_set = list(ParameterGrid(model_kwargs))

        results = []
        for param in tqdm(param_set):
            model = model_class(**param)

            try:
                model.fit(x_trains[0], y_trains[0])
            except ValueError:
                pass

            if len(unique_labels) == 1:
                pred = [unique_labels[0]] * len(x_valid)
                pred_prob = np.array([1.] * len(x_valid))
            else:
                pred = model.predict(x_valid)
                pred_prob = model.predict_proba(x_valid)

            binary_f1 = f1_score(y_valid, pred, average='binary')
            weighted_f1 = _weighted_f1(y_valid, pred)
            acc = accuracy_score(y_valid, pred)
            precision = precision_score(y_valid, pred, average='binary')
            recall = recall_score(y_valid, pred, average='binary')
            macro_f1 = f1_score(y_valid, pred, average='macro')

            # auroc
            size = [a["size"] for a in metadata["columns"] if a["name"] == "label"][0]
            rest_label = set(range(size)) - set(unique_labels)
            tmp = []
            j = 0
            for i in range(size):
                if i in rest_label:
                    tmp.append(np.array([0] * y_valid.shape[0])[:,np.newaxis])
                else:
                    try:
                        tmp.append(pred_prob[:,[j]])
                    except:
                        tmp.append(pred_prob[:, np.newaxis])
                    j += 1
            try:
                roc_auc = roc_auc_score(np.eye(size)[y_valid], np.hstack(tmp))
            except:
                roc_auc = -1
            results.append(
                {   
                    "name": model_repr,
                    "param": param,
                    "binary_f1": binary_f1,
                    "weighted_f1": weighted_f1,
                    "roc_auc": roc_auc, 
                    "accuracy": acc, 
                    "precision": precision, 
                    "recall": recall, 
                    "macro_f1": macro_f1
                }
            )

        # test the best model
        results = pd.DataFrame(results)
        results['avg'] = results.loc[:, ['binary_f1', 'weighted_f1', 'roc_auc']].mean(axis=1)        
        best_f1_param = results.param[results.binary_f1.idxmax()]
        best_weighted_param = results.param[results.weighted_f1.idxmax()]
        best_auroc_param = results.param[results.roc_auc.idxmax()]
        best_acc_param = results.param[results.accuracy.idxmax()]
        best_avg_param = results.param[results.avg.idxmax()]

        def _calc(best_model):
            best_scores = []
            for x_train, y_train in zip(x_trains, y_trains):
                try:
                    best_model.fit(x_train, y_train)
                except ValueError:
                    pass

                if len(unique_labels) == 1:
                    pred = [unique_labels[0]] * len(x_test)
                    pred_prob = np.array([1.] * len(x_test))
                else:
                    pred = best_model.predict(x_test)
                    pred_prob = best_model.predict_proba(x_test)

                binary_f1 = f1_score(y_test, pred, average='binary')
                weighted_f1 = _weighted_f1(y_test, pred)
                acc = accuracy_score(y_test, pred)
                precision = precision_score(y_test, pred, average='binary')
                recall = recall_score(y_test, pred, average='binary')
                macro_f1 = f1_score(y_test, pred, average='macro')

                # auroc
                size = [a["size"] for a in metadata["columns"] if a["name"] == "label"][0]
                rest_label = set(range(size)) - set(unique_labels)
                tmp = []
                j = 0
                for i in range(size):
                    if i in rest_label:
                        tmp.append(np.array([0] * y_test.shape[0])[:,np.newaxis])
                    else:
                        try:
                            tmp.append(pred_prob[:,[j]])
                        except:
                            tmp.append(pred_prob[:, np.newaxis])
                        j += 1
                # try:
                #     roc_auc = roc_auc_score(np.eye(size)[y_test], np.hstack(tmp))
                # except ValueError:
                #     tmp[1] = tmp[1].reshape(20000, 1)
                #     roc_auc = roc_auc_score(np.eye(size)[y_test], np.hstack(tmp))
                #     # roc_auc = -1

                best_scores.append(
                    {   
                        # "name": model_repr,
                        # "param": param,
                        "binary_f1": binary_f1,
                        "weighted_f1": weighted_f1,
                        # "roc_auc": roc_auc,
                        "accuracy": acc, 
                        "precision": precision, 
                        "recall": recall, 
                        "macro_f1": macro_f1
                    }
                )
                # print(len(best_scores))
            return pd.DataFrame(best_scores).mean(axis=0)

        def _df(dataframe):
            return {
                "name": model_repr,
                # "roc_auc": dataframe.roc_auc,
                # "weighted_f1": dataframe.weighted_f1,
                "down_accuracy": dataframe.accuracy,
                "down_precision":dataframe.precision,
                "down_recall":dataframe.recall,
                "binary_f1": dataframe.binary_f1,
            }

        best_f1_scores.append(_df(_calc(model_class(**best_f1_param))))
        best_weighted_scores.append(_df(_calc(model_class(**best_weighted_param))))
        best_auroc_scores.append(_df(_calc(model_class(**best_auroc_param))))
        best_acc_scores.append(_df(_calc(model_class(**best_acc_param))))
        best_avg_scores.append(_df(_calc(model_class(**best_avg_param))))

     

    return pd.DataFrame(best_f1_scores), pd.DataFrame(best_weighted_scores), pd.DataFrame(best_auroc_scores)


@ignore_warnings(category=ConvergenceWarning)
def _evaluate_regression(train, test, fake, metadata):
    x_trains, y_trains, x_valid, y_valid, x_test, y_test, regressors = _prepare_ml_problem(fake, train, test, metadata)

    best_r2_scores = []
    best_ev_scores = []
    best_mae_scores = []
    best_rmse_scores = []
    best_avg_scores = []

    y_trains = [np.log(np.clip(i, 1, 20000)) for i in y_trains]
    y_test = np.log(np.clip(y_test, 1, 20000))

    for model_spec in regressors:
        model_class = model_spec['class']
        model_kwargs = model_spec.get('kwargs', dict())
        model_repr = model_class.__name__

        param_set = list(ParameterGrid(model_kwargs))

        results = []
        for param in tqdm(param_set):
            model = model_class(**param)
            model.fit(x_trains[0], y_trains[0])
            pred = model.predict(x_valid)

            r2 = r2_score(y_valid, pred)
            explained_variance = explained_variance_score(y_valid, pred)
            mean_squared = mean_squared_error(y_valid, pred)
            root_mean_squared = mean_squared_error(y_valid, pred, squared=False)
            mean_absolute = mean_absolute_error(y_valid, pred)

            results.append(
                {   
                    "name": model_repr,
                    "param": param,
                    "r2": r2,
                    "explained_variance": explained_variance,
                    "mean_squared": mean_squared, 
                    "mean_absolute": mean_absolute, 
                    "rmse": root_mean_squared
                }
            )

        results = pd.DataFrame(results)
        # results['avg'] = results.loc[:, ['r2', 'rmse']].mean(axis=1)
        best_r2_param = results.param[results.r2.idxmax()]
        best_ev_param = results.param[results.explained_variance.idxmax()]
        best_mae_param = results.param[results.mean_absolute.idxmin()]
        best_rmse_param = results.param[results.rmse.idxmin()]
        # best_avg_param = results.param[results.avg.idxmax()]

        def _calc(best_model):
            best_scores = []
            for x_train, y_train in zip(x_trains, y_trains):
                best_model.fit(x_train, y_train)
                pred = best_model.predict(x_test)

                r2 = r2_score(y_test, pred)
                explained_variance = explained_variance_score(y_test, pred)
                mean_squared = mean_squared_error(y_test, pred)
                root_mean_squared = mean_squared_error(y_test, pred, squared=False)
                mean_absolute = mean_absolute_error(y_test, pred)

                best_scores.append(
                    {   
                        # "name": model_repr,
                        "param": param,
                        "r2": r2,
                        "explained_variance": explained_variance,
                        "mean_squared": mean_squared, 
                        "mean_absolute": mean_absolute, 
                        "rmse": root_mean_squared
                    }
                )

            return pd.DataFrame(best_scores).mean(axis=0)

        def _df(dataframe):
            return {
                "name": model_repr,
                "r2": dataframe.r2,
                "explained_variance": dataframe.explained_variance,
                "MAE": dataframe.mean_absolute,
                "RMSE": dataframe.rmse,
            }

        best_r2_scores.append(_df(_calc(model_class(**best_r2_param))))
        best_ev_scores.append(_df(_calc(model_class(**best_ev_param))))
        best_mae_scores.append(_df(_calc(model_class(**best_mae_param))))
        best_rmse_scores.append(_df(_calc(model_class(**best_rmse_param))))

    return pd.DataFrame(best_r2_scores), pd.DataFrame(best_rmse_scores), None



_EVALUATORS = {
    'binary_classification': _evaluate_binary_classification,
    'multiclass_classification': _evaluate_multi_classification,
    'regression': _evaluate_regression
}



def compute_scores(train, test, synthesized_data, metadata):
    global INPUT_SIZE
    INPUT_SIZE = len(metadata['columns'])-1
    global _MODELS
    _MODELS = default_models()
    a, b, c = _EVALUATORS[metadata['problem_type']](train=train, test=test, fake=synthesized_data, metadata=metadata)
    # return a.drop(columns=['name']).mean(axis=0), a.drop(columns=['name']).std(axis=0)
    return c, c
