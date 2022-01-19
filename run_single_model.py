from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier


def build_dataset(_data_rs, _sample_rs):
    x, y = make_classification(
        n_samples=1000, n_features=50, n_informative=12, n_redundant=12, random_state=_data_rs
    )

    train_samples = 800  # Samples used for training the models
    _x_train, _x_test, _y_train, _y_test = train_test_split(
        x,
        y,
        shuffle=False,
        test_size=1000 - train_samples,
        random_state=_sample_rs
    )
    return _x_train, _x_test, _y_train, _y_test


def train(_mod, _x_train, _x_test, _y_train, _y_test, _rs):
    if _mod == 'RandomForest':
        _clf = RandomForestClassifier(n_estimators=100, random_state=_rs)
    elif _mod == 'SVC':
        _clf = SVC(random_state=_rs)
    elif _mod == 'NeuralNet':
        _clf = MLPClassifier(random_state=_rs)
    elif _mod == 'NaiveBayes':
        _clf = GaussianNB()
    elif _mod == 'LogisticRegression':
        _clf = LogisticRegression(random_state=_rs)
    elif _mod == 'AdaBoost':
        _clf = AdaBoostClassifier(random_state=_rs)
    _clf.fit(_x_train, _y_train)
    _preds = _clf.predict(_x_test)
    _score = f1_score(_y_test, _preds)
    # print('Random state {}, model performance {}'.format(_rs, _score))
    return _score


def run(_mod, _data_rs, _sample_rs, _model_rs):
    x_train, x_test, y_train, y_test = build_dataset(_data_rs, _sample_rs)
    _score = train(_mod, x_train, x_test, y_train, y_test, _model_rs)
    return _score

if __name__ == "__main__":
    data_rs = 17
    sample_rs = 17
    model_rs = 17
    run(data_rs, sample_rs, model_rs)