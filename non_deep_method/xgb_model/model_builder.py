from lightgbm import LGBMClassifier, early_stopping, log_evaluation

from non_deep_method.data_builder.xy_data_builder import data_builder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


class ModelBuilder:
    def __init__(self, top_n):
        self.clf = LGBMClassifier(n_jobs=4,
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
        self.X_train, self.y_train = data_builder.build_data_with_features(top_n=top_n, phase='train_phase',
                                                                           prefix='train_ques')
        self.X_test, self.y_test = data_builder.build_data_with_features(top_n=top_n, phase='test_phase',
                                                                         prefix='train_ques')

    def test(self):
        pred_prob = self.clf.predict_proba(X=self.X_test)
        pred_prob = pred_prob[:, 1]
        print(roc_auc_score(y_true=self.y_test, y_score=pred_prob))

    def start_train(self):
        x_train, x_val, y_train, y_val = train_test_split(self.X_train, self.y_train, test_size=0.2)
        self.clf.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_val, y_val)],
                     callbacks=[early_stopping(stopping_rounds=200), log_evaluation(period=200)],
                     eval_metric='auc')


model_builder = ModelBuilder(top_n=50)
model_builder.start_train()

if __name__ == '__main__':
    model_builder.test()
