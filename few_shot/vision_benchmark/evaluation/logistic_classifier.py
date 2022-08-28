"""
Classifier with sklearn logistic regression (lbfgs solver).
"""
import time
import logging
import multiprocessing
import numpy as np
from sklearn.linear_model import LogisticRegression
import sharedmem
from .metric import get_metric


def calculate_val(idx, train_features, train_labels, val_features, val_labels, l2_lambda_list, val_accuracies, metric, max_iter):
    classifier = LogisticRegression(C=l2_lambda_list[idx], max_iter=max_iter, verbose=0, warm_start=True, n_jobs=1)
    classifier.fit(train_features, train_labels)
    # Validation score
    val_predict = classifier.predict_proba(val_features)
    val_accuracy = metric(val_labels, val_predict)
    val_accuracies[idx] = val_accuracy


def hyperparameter_sweep(train_features, train_labels, val_features, val_labels, metric):
    """
    Hyper parameter sweep for l2 regularization strength.
    Refer to https://arxiv.org/pdf/2103.00020.pdf Page 38 A.3 Evaluation for details.
    """
    logging.info("=> Start hyperparameter tuning.")
    start = time.time()
    MAX_ITERATION = 1000
    l2_lambda_list = np.logspace(-6, 6, num=97).tolist()
    val_accuracies = [None] * len(l2_lambda_list)
    iter_num = 0

    train_features_size, train_labels_size, val_features_size, val_labels_size = 1, 1, 1, 1
    for dim in train_features.shape:
        train_features_size *= dim
    for dim in train_labels.shape:
        train_labels_size *= dim
    for dim in val_features.shape:
        val_features_size *= dim
    for dim in val_labels.shape:
        val_labels_size *= dim
    train_features_share = sharedmem.empty(train_features_size, dtype=np.float)
    train_labels_share = sharedmem.empty(train_labels_size, dtype=np.float)
    val_features_share = sharedmem.empty(val_features_size, dtype=np.float)
    val_labels_share = sharedmem.empty(val_labels_size, dtype=np.float)
    train_features_share = train_features
    train_labels_share = train_labels
    val_features_share = val_features
    val_labels_share = val_labels

    with multiprocessing.Manager() as manager:
        l2_lambda_list = manager.list(l2_lambda_list)
        val_accuracies = manager.list(val_accuracies)
        l2_lambda_init_idx = [i for i, val in enumerate(l2_lambda_list) if val in set(np.logspace(-6, 6, num=7))]

        def find_peak(l2_lambda_idx):
            jobs = []
            for idx in l2_lambda_idx:
                if not val_accuracies[idx]:
                    p = multiprocessing.Process(
                        target=calculate_val,
                        args=(idx, train_features_share, train_labels_share, val_features_share, val_labels_share, l2_lambda_list, val_accuracies, metric, MAX_ITERATION))
                    jobs.append(p)
                    p.start()
            for proc in jobs:
                proc.join()
            peak_idx = -1
            peak_metric_score = 0
            for idx in l2_lambda_idx:
                l2_lambda_inv = l2_lambda_list[idx]
                if val_accuracies[idx] > peak_metric_score:
                    peak_idx = idx
                    peak_metric_score = val_accuracies[idx]
                logging.info(f"=>Search index: {idx}, L2_lambda_inv: {l2_lambda_inv}, Val {metric.__name__}: {val_accuracies[idx]}")
            logging.info(f"=>Iter. {iter_num}: Best lambda inv index: {peak_idx}, L2_lambda_inv: {l2_lambda_list[peak_idx]}, Val {metric.__name__}: {peak_metric_score}")
            return peak_idx

        l2_lambda_init_idx = [i for i, val in enumerate(l2_lambda_list) if val in set(np.logspace(-6, 6, num=7))]
        peak_idx = find_peak(l2_lambda_init_idx)
        step_span = 8
        while step_span > 0:
            left, right = max(peak_idx - step_span, 0), min(peak_idx + step_span, len(l2_lambda_list)-1)
            iter_num += 1
            peak_idx = find_peak([left, peak_idx, right])
            step_span //= 2

        logging.info(f"The best l2 lambda inverse is {l2_lambda_list[peak_idx]}")
        logging.info('=> Hyperparameter tuning duration time: {:.2f}s'.format(time.time()-start))
        logging.info('=> Finished hyperparameter tuning.')
        return l2_lambda_list[peak_idx]


def lr_classifier(train_features, train_labels, val_features, val_labels, test_features, test_labels, no_hyperparameter_tuning, l2inv, config):
    """
    logistic regression classifier with sklearn's L-BFGS implementation.
    """
    metric = get_metric(config.TEST.METRIC)
    # Hyperparameter sweep
    if no_hyperparameter_tuning:
        l2_lambda_inv = l2inv
    else:
        l2_lambda_inv = hyperparameter_sweep(train_features, train_labels, val_features, val_labels, metric)
        # l2_lambda_inv = hyperparameter_sweep(np.concatenate((train_features, val_features)), np.concatenate((train_labels,val_labels)), test_features, test_labels, metric)
    # Combine training and validation sets together to train the final classifier
    logging.info("=> The final classifier is on training ...")
    logging.info(f"Hyperparameters: l2_lambda_inv = {l2_lambda_inv}")
    train_features_all = np.concatenate((train_features, val_features))
    train_labels_all = np.concatenate((train_labels, val_labels))
    classifier = LogisticRegression(C=l2_lambda_inv, max_iter=1000, verbose=0)
    classifier.fit(train_features_all, train_labels_all)

    # Test on the testing set.
    test_pred = classifier.predict_proba(test_features)
    metric_score = metric(test_labels, test_pred)
    logging.info(f"Test score: {metric.__name__} = {metric_score:.3f}")
    return test_pred
