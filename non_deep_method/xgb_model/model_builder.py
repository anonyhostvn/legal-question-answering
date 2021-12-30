from lightgbm import LGBMClassifier, early_stopping, log_evaluation

from non_deep_method.data_builder.xy_data_builder import data_builder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np


class ModelBuilder:
    def __init__(self):
        self.clf = []

    def test(self, X, y):
        # [n_clf * n_sample]
        pred_prob = np.array([single_clf.predict_proba(X=X)[:, 1] for single_clf in self.clf])

        pred_prob = np.mean(pred_prob, axis=0)

        print('roc-auc score one test set: ', roc_auc_score(y_true=y, y_score=pred_prob))

    def train_single_model(self, X, y):
        self.clf = []
        print('Start training single xgb model')
        x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        single_clf = LGBMClassifier(n_jobs=4,
                                    n_estimators=4000,
                                    learning_rate=0.02,
                                    num_leaves=34,
                                    colsample_bytree=0.9497036,
                                    subsample=0.8715623,
                                    max_depth=8,
                                    reg_alpha=0.041545473,
                                    reg_lambda=0.0735294,
                                    min_split_gain=0.0222415,
                                    min_child_weight=39.3259775,
                                    verbose=-1
                                    )
        single_clf.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_val, y_val)],
                       callbacks=[early_stopping(stopping_rounds=200), log_evaluation(period=200)],
                       eval_metric='auc')
        self.clf.append(single_clf)

    def train_k_fold(self, X, y):
        skf = StratifiedKFold(n_splits=6)
        self.clf = []
        for fold_id, (train_index, val_index) in enumerate(skf.split(X=X, y=y)):
            print('Start training fold: ', fold_id)
            single_clf = LGBMClassifier(n_jobs=4,
                                        n_estimators=4000,
                                        learning_rate=0.02,
                                        num_leaves=34,
                                        colsample_bytree=0.9497036,
                                        subsample=0.8715623,
                                        max_depth=8,
                                        reg_alpha=0.041545473,
                                        reg_lambda=0.0735294,
                                        min_split_gain=0.0222415,
                                        min_child_weight=39.3259775,
                                        verbose=-1
                                        )
            x_train = X[train_index]
            x_val = X[val_index]
            y_train = y[train_index]
            y_val = y[val_index]
            single_clf.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_val, y_val)],
                           callbacks=[early_stopping(stopping_rounds=200), log_evaluation(period=400)],
                           eval_metric='auc')
            self.clf.append(single_clf)


if __name__ == '__main__':
    top_n = 50
    train_x, train_y = data_builder.build_data_with_features(top_n=top_n, phase='train_phase',
                                                             prefix='train_ques')
    test_x, test_y = data_builder.build_data_with_features(top_n=top_n, phase='test_phase',
                                                           prefix='train_ques')
    model_builder = ModelBuilder()

    # model_builder.train_single_model(X=train_x, y=train_y)
    model_builder.train_k_fold(X=train_x, y=train_y)

    model_builder.test(X=test_x, y=test_y)
