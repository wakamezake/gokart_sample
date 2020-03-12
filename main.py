import gokart
import lightgbm as lgb
import luigi
import numpy as np
from sklearn.model_selection import train_test_split


def model_train(x_train, y_train, x_valid, y_valid):
    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_eval = lgb.Dataset(x_valid, y_valid, reference=lgb_train)

    params = {
        'objective': 'binary'
    }

    model = lgb.train(
        params, lgb_train,
        valid_sets=[lgb_train, lgb_eval],
        verbose_eval=10,
        num_boost_round=1000,
        early_stopping_rounds=10
    )
    return model


def preprocess(data):
    data['Sex'].replace(['male', 'female'], [0, 1], inplace=True)
    data['Embarked'].fillna('S', inplace=True)
    data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(
        int)
    data['Fare'].fillna(np.mean(data['Fare']), inplace=True)
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['FamilySize'] = data['Parch'] + data['SibSp'] + 1
    data['IsAlone'] = 0
    data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1
    return data


class DataTask(gokart.TaskOnKart):
    _path = "https://raw.githubusercontent.com/pcsanwald/kaggle-titanic/master/"
    _train_path = "{}{}".format(_path, "train.csv")
    _test_path = "{}{}".format(_path, "test.csv")
    train_path = luigi.Parameter(default=_train_path)
    test_path = luigi.Parameter(default=_test_path)

    def output(self):
        return self.make_target("resources/train.pkl")

    def run(self):
        self.dump(self.train_path)


class FeatureTask(gokart.TaskOnKart):
    _categorical_features = ['Embarked', 'Pclass', 'Sex']
    categorical_features = luigi.Parameter(default=_categorical_features)
    _target_col = "Survived"
    target_col = luigi.Parameter(default=_target_col)

    def requires(self):
        return DataTask()

    def output(self):
        return self.make_target("resources/feature.pkl")

    def run(self):
        train = self.load_data_frame()
        train.drop(columns=self.target_col, inplace=True)
        self.dump(preprocess(train))


class TargetTask(gokart.TaskOnKart):
    _target_col = "Survived"
    target_col = luigi.Parameter(default=_target_col)

    def requires(self):
        return DataTask()

    def output(self):
        return self.make_target("resources/feature.pkl")

    def run(self):
        train = self.load_data_frame()
        target = train[self.target_col]
        self.dump(target)


class TrainTask(gokart.TaskOnKart):
    def requires(self):
        return dict(features=FeatureTask(), target=TargetTask())

    def run(self):
        features = self.load_data_frame("features")
        target = self.load_data_frame("target")

        model_train(**train_test_split(features, target))


if __name__ == '__main__':
    luigi.run(TrainTask())
