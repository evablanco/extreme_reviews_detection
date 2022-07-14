import numpy as np
import pandas as pd
import sklearn
from nltk.tokenize import word_tokenize

# Kaggle paths
TRAIN_PATH = './data/kaggle_train.csv'
VAL_PATH = './data/kaggle_val.csv'
TEST_PATH = './data/kaggle_test.csv'


def int_converter(label):
    return int(label)


def set_binary_labels(label):
    if (label > 3):
        return 1
    else:
        return 0


def set_sentiment_labels(label):
    if (label == 3):
        return 1
    elif (label > 3):
        return 2
    else:
        return 0


def set_five_labels(label):
    if (label == 1):
        return 0
    elif (label == 2):
        return 1
    elif (label == 3):
        return 2
    elif label == 4:
        return 3
    else:
        return 4


def set_extreme_labels(label):
    if label == 5:
        return 1
    else:
        return 0


def drop_revs(df_orig, max_len):
    df = df_orig.copy()
    print('No. of reviews in dataset = ', df.shape[0])
    df_aux_nan = df[df.review_full == np.nan]
    df_orig.drop(df_aux_nan.index, inplace=True)
    df['words'] = df['review_full'].apply(word_tokenize)
    df['word_count'] = df['words'].apply(lambda x: len(x))
    df_aux = df[df.word_count > max_len]
    num_rows_orig = df_orig.shape[0]
    df_orig.drop(df_aux.index, inplace=True)
    num_rows_filtered = df_orig.shape[0]
    print('No. of reviews in dataset with less than ', max_len, ' words = ', df.shape[0])
    print('Removed ', num_rows_orig - num_rows_filtered, ' examples.')
    return df_orig


def read_data_from_path(max_len):
    train_set = pd.read_csv(TRAIN_PATH)
    val_set = pd.read_csv(VAL_PATH)
    test_set = pd.read_csv(TEST_PATH)
    # Convert labels to int
    train_set['rating'] = train_set['rating'].apply(int_converter)
    val_set['rating'] = val_set['rating'].apply(int_converter)
    test_set['rating'] = test_set['rating'].apply(int_converter)
    # Remove reviews with more than max_len words
    train_set = drop_revs(train_set, max_len)
    val_set = drop_revs(val_set, max_len)
    test_set = drop_revs(test_set, max_len)
    return train_set, val_set, test_set


def set_labels_fun(num_classes):
    if num_classes == 2:
        fun = set_binary_labels
    elif num_classes == 5:
        fun = set_five_labels
    else:
        print('Not supported')
        fun = None
    return fun


def get_data_val(num_classes, max_len):
    fun = set_labels_fun(num_classes)
    train_set, val_set, test_set = read_data_from_path(max_len)
    train_set['rating'] = train_set['rating'].apply(fun)
    val_set['rating'] = val_set['rating'].apply(fun)
    test_set['rating'] = test_set['rating'].apply(fun)
    return train_set, val_set, test_set


def get_data(num_classes, max_len):
    fun = set_labels_fun(num_classes)
    train_set, val_set, test_set = read_data_from_path(max_len)
    train_set['rating'] = train_set['rating'].apply(fun)
    val_set['rating'] = val_set['rating'].apply(fun)
    test_set['rating'] = test_set['rating'].apply(fun)
    all_train = train_set.copy()
    all_train = all_train.append(val_set)
    return all_train, test_set


def get_extremes(df):
    selected_df_1s = df[df['rating'] == 1]
    selected_df_5s = df[df['rating'] == 5]
    final_df = selected_df_1s.append(selected_df_5s)
    final_df = sklearn.utils.shuffle(final_df)
    final_df['rating'] = final_df['rating'].apply(set_extreme_labels)
    return final_df


def get_extremes_balanced(df, train=False):
    selected_df_1s = df[df['rating'] == 1]
    selected_df_5s = df[df['rating'] == 5]
    if train:
        final_df = selected_df_1s.append(selected_df_5s[0:len(selected_df_1s)])
    else:
        final_df = selected_df_1s.append(selected_df_5s)
    final_df = sklearn.utils.shuffle(final_df)
    final_df['rating'] = final_df['rating'].apply(set_extreme_labels)
    return final_df


def get_data_extremes_val(max_len):
    train_set, val_set, test_set = read_data_from_path(max_len)
    train_set = get_extremes(train_set)
    val_set = get_extremes(val_set)
    test_set = get_extremes(test_set)
    return train_set, val_set, test_set

def get_data_extremes_balanced_val(max_len):
    train_set, val_set, test_set = read_data_from_path(max_len)
    train_set = get_extremes_balanced(train_set, True)
    val_set = get_extremes_balanced(val_set)
    test_set = get_extremes_balanced(test_set)
    return train_set, val_set, test_set


def get_data_extremes(max_len):
    train_set, val_set, test_set = read_data_from_path(max_len)
    train_set = get_extremes(train_set)
    val_set = get_extremes(val_set)
    test_set = get_extremes(test_set)
    all_train = train_set.copy()
    all_train = all_train.append(val_set)
    return all_train, test_set


def get_data_extremes_balanced(max_len):
    train_set, val_set, test_set = read_data_from_path(max_len)
    train_set = get_extremes_balanced(train_set, True)
    val_set = get_extremes_balanced(val_set, True)
    test_set = get_extremes_balanced(test_set)
    all_train = train_set.copy()
    all_train = all_train.append(val_set)
    return all_train, test_set


def get_toy_data(num_classes, max_len):
    fun = set_labels_fun(num_classes)
    train_set, val_set, test_set = read_data_from_path(max_len)
    train_set['rating'] = train_set['rating'].apply(fun)
    val_set['rating'] = val_set['rating'].apply(fun)
    test_set['rating'] = test_set['rating'].apply(fun)
    all_train = train_set.copy()
    all_train = all_train.append(val_set)
    return all_train[0:1000], test_set[0:500]


def get_toy_data_val(num_classes, max_len):
    fun = set_labels_fun(num_classes)
    train_set, val_set, test_set = read_data_from_path(max_len)
    train_set['rating'] = train_set['rating'].apply(fun)
    val_set['rating'] = val_set['rating'].apply(fun)
    test_set['rating'] = test_set['rating'].apply(fun)
    return train_set[0:1000], val_set[0:150], test_set[0:500]

