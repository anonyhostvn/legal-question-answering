from lightgbm import LGBMClassifier, early_stopping, log_evaluation

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np
import pickle
import os


class ModelBuilder:
    def __init__(self, lis_clf=None):
        if lis_clf is None:
            self.clf = []
        else:
            self.clf = lis_clf
        self.SAVE_LIS_MODEL_PATH = os.path.join('non_deep_method', 'cached', 'lis_xgb.pkl')

    def predict_prob(self, X):
        pred_prob = np.array([single_clf.predict_proba(X=X)[:, 1] for single_clf in self.clf])
        pred_prob = np.mean(pred_prob, axis=0)
        return pred_prob

    def test(self, X, y):
        # [n_clf * n_sample]
        pred_prob = self.predict_prob(X=X)
        print('roc-auc score one test set: ', roc_auc_score(y_true=y, y_score=pred_prob))

    def base_model(self):
        return LGBMClassifier(n_jobs=4,
                              n_estimators=8192,
                              learning_rate=0.02,
                              colsample_bytree=0.9,
                              subsample=0.9,
                              # max_depth=16,
                              # reg_alpha=0.041545473,
                              # reg_lambda=0.0735294,
                              # min_split_gain=0.0222415,
                              # min_child_weight=39.3259775,
                              verbose=-1,
                              scale_pos_weight=5,
                              # num_iterations=1024,
                              num_leaves=256,
                              max_bins=512
                              )

    def train_single_model(self, X, y):
        self.clf = []
        print('Start training single xgb model')
        x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        single_clf = self.base_model()
        single_clf.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_val, y_val)],
                       callbacks=[early_stopping(stopping_rounds=200), log_evaluation(period=100)],
                       eval_metric='auc')
        self.clf.append(single_clf)

    def train_k_fold(self, X, y):
        skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)
        self.clf = []
        for fold_id, (train_index, val_index) in enumerate(skf.split(X=X, y=y)):
            single_clf = self.base_model()
            print(val_index)
            x_train = X[train_index]
            x_val = X[val_index]
            y_train = y[train_index]
            y_val = y[val_index]
            print('Start training fold: ', fold_id)
            print(' number of positive training examples: ', sum(y_train))
            print('total number of validation examples: ', len(x_val))
            print('number of positive validation examples: ', sum(y_val))
            single_clf.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_val, y_val)],
                           callbacks=[early_stopping(stopping_rounds=200), log_evaluation(period=100)],
                           eval_metric='auc')
            self.clf.append(single_clf)
        pickle.dump(self.clf, open(self.SAVE_LIS_MODEL_PATH, 'wb'))
