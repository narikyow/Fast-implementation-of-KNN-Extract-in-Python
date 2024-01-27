import numpy as np
from sklearn.model_selection import StratifiedKFold
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from numba import jit

@jit(nopython=True)
def _distance(a, b):
    return np.sum(np.abs(a-b), axis=1)

@jit(nopython=True)
def _get_feat(data, X_train, k_index):
    distances = np.sum(np.abs(X_train-data), axis=1)
    sorted_distances_index = np.argsort(distances)
    nearest_index = sorted_distances_index[0: (k_index + 1)]
    return np.sum(distances[nearest_index])
    
def parallel_get_feat_train(args):
    train_index, valid_index, class_index, k_index, X, y = args
    print("--TRAIN--")
    arg_1 = X.iloc[list(valid_index)]#.values.astype(float)
    arg_2 = X.iloc[list(train_index)][y.iloc[list(train_index)]==class_index].values.astype(float)
    feat_train = np.array([np.apply_along_axis(
    _get_feat, 1,
    arg_1, arg_2, k_index
    )])
    print("TRAIN_DONE")
    return feat_train

def parallel_get_feat_test(args):
    train_index, valid_index, class_index, k_index, X, y, X_test = args
    print("--TEST--")
    arg_1 = X_test.values.astype(float)
    arg_2 = X.iloc[list(train_index)][y.iloc[list(train_index)]==class_index].values.astype(float)
    feat_test = np.array([np.apply_along_axis(
    _get_feat, 1,
    arg_1, arg_2, k_index
    )])
    print("TEST_DONE")
    return feat_test

def knnExtract(X, y, X_test, skf, k=3):# ターゲットエンコーディング用関数

    CLASS_NUM = len(set(y))
    res_train = np.empty((X.shape[0], CLASS_NUM * k))
    res_test = np.empty((X_test.shape[0], CLASS_NUM * k))
    col_names = []
    # まず、X_testのターゲットエンコーディングを全ての学習データを用いて行う。
    # X_testの目的変数は不明なため、X_trainのすべてのデータをエンコーディングに用いる。
    with ThreadPoolExecutor(max_workers=None) as executor:
        args_train = []
        args_test = []
        for train_index, valid_index in skf.split(X, y):
            X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
            y_train          = y.iloc[train_index]
            features_train = np.empty([0, X_valid.shape[0]])
            features_test = np.empty([0, X_test.shape[0]])
            for class_index in range(CLASS_NUM):
                for k_index in range(k):
                    print(f"knnExtract_{class_index}_{k_index}")
                    args_train.append([tuple(train_index), tuple(valid_index), class_index, k_index])
                    args_test.append([tuple(train_index), tuple(valid_index), class_index, k_index])
                    col_names.append(f"prob_knn_{class_index}_{k_index}")
        col_names = list(set(col_names))
        future_train = {tuple(i): executor.submit(parallel_get_feat_train, (i[0], i[1], i[2], i[3], X, y)) for i in args_train}
        future_test = {tuple(i): executor.submit(parallel_get_feat_test, (i[0], i[1], i[2], i[3], X, y, X_test)) for i in args_test}
        for train_index, valid_index in skf.split(X, y):
            X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
            y_train          = y.iloc[train_index]
            features_train = np.empty([0, X_valid.shape[0]])
            features_test = np.empty([0, X_test.shape[0]])
            for class_index in range(CLASS_NUM):
                for k_index in range(k):
                    # apply_along_axis: 配列に関数を適用する。 axis=0なら各列、1なら各行
                    print(f"knnExtract_{class_index}_{k_index}")
                    arg_train = (tuple(train_index), tuple(valid_index), class_index, k_index)
                    arg_test = (tuple(train_index), tuple(valid_index), class_index, k_index)
                    features_train = np.append(features_train, future_train[tuple(arg_train)].result(), axis=0)
                    features_test = np.append(features_test, future_test[tuple(arg_test)].result(), axis=0)
            print(features_train)
            print(features_test)
            res_train[valid_index] = features_train.T
            res_test += features_test.T # CV毎に加算
        res_test /= skf.get_n_splits()
    
    res_knn_train_df = pd.DataFrame(res_train, index=X.index, columns=col_names)
    res_knn_test_df = pd.DataFrame(res_test, index=X_test.index, columns=col_names)

    return res_knn_train_df, res_knn_test_df


def knn_SKF(X, y, X_test, folds=7): # StKfoldCVをした後、ターゲットエンコーディングする関数。
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    col = X.columns
    df = pd.DataFrame()
    df[col] = scaler.fit_transform(X)
    df.index = X.index
    X = df.copy()

    col = X_test.columns
    df = pd.DataFrame()
    df[col] = scaler.transform(X_test)
    df.index = X_test.index
    
    X_test = df.copy()

    train_knn, test_knn = knnExtract(X.select_dtypes("number"), y, X_test.select_dtypes("number"), skf, k=3)

    X = pd.concat([X, train_knn], axis=1)
    X_test = pd.concat([X_test, test_knn], axis=1)

    return X, X_test
