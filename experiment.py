import numpy as np
import pandas as pd

from tqdm import tqdm

from run_single_model import run
from bootstrap_significance import bootci_diff, bootpv

from warnings import filterwarnings
filterwarnings('ignore')


def train_random_models(_mod, _n_models):
    batch = int(_n_models / 2)
    _scores_a = []
    _scores_b = []
    batch_a_data_rs = [np.random.randint(1, 100) for k in range(batch)]
    batch_a_sample_rs = [np.random.randint(1, 100) for k in range(batch)]
    batch_a_model_rs = [np.random.randint(1, 100) for k in range(batch)]
    batch_b_data_rs = [np.random.randint(1, 100) for k in range(batch)]
    batch_b_sample_rs = [np.random.randint(1, 100) for k in range(batch)]
    batch_b_model_rs = [np.random.randint(1, 100) for k in range(batch)]
    for j in tqdm(range(batch)):
        cur_s = run(_mod,
                    batch_a_data_rs[j],
                    batch_a_sample_rs[j],
                    batch_a_model_rs[j])
        _scores_a.append(cur_s)
        if j == 0:
            print('F1 was {}'.format(cur_s))
        _scores_b.append(run(_mod,
                             batch_b_data_rs[j],
                             batch_b_sample_rs[j],
                             batch_b_model_rs[j]))
    return _scores_a, _scores_b

def eval_models(_a, _a_hat):
    ci, bootd = bootci_diff(_a, _a_hat, keepboot=True)
    print(ci)
    print(bootd)
    _pv = bootpv(_a, _a_hat)
    print(_pv)
    return _pv


def eval_best(_a, _a_hat):
    a_star = np.max(_a)
    a_hat_star = np.max(_a_hat)
    ci, bootd = bootci_diff(a_star, a_hat_star, keepboot=True)
    print(ci)
    print(bootd)
    _pv = bootpv(a_star, a_hat_star)
    print(_pv)
    return _pv


def compute_threshold_difference(_scores_a, _scores_b):
    f1_diff = []
    for j in range(len(_scores_a)):
        f1_diff.append(np.abs(_scores_a[j] - _scores_b[j]))
    _tau = np.mean(f1_diff)
    return _tau

def run_task(_mod, _num_mod):
    a, a_hat = train_random_models(_mod, _num_mod)
    pct_all = eval_models(a, a_hat)
    _tau = compute_threshold_difference(a, a_hat)
    #pct_best = eval_best(a, a_hat)
    return _tau, pct_all #, pct_best


if __name__ == "__main__":
    scores_all = []
    scores_best = []
    taus = []
    available_models = ['AdaBoost',
                        'RandomForest',
                        #'SVC',
                        'NeuralNet',
                        #'NaiveBayes',
                        ]
    for m in available_models:
        tau, sa = run_task(m, 1000)
        scores_all.append(sa)
        #scores_best.append(sb)
        taus.append(tau)
        print(tau)
    results = pd.DataFrame({'models': available_models,
                            'eval1': scores_all,
                            #'eval2': scores_best,
                            'tau': taus})
    results.to_csv('single_performance_scores.csv', index=False)