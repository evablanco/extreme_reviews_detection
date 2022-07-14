import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
import random
import os
import time
from transformers import XLNetForSequenceClassification
from transformers import XLNetTokenizer, XLNetModel
from transformers import AdamW
import torch
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from transformers import TFXLNetModel, XLNetTokenizer, XLNetConfig
from tensorflow.keras.models import save_model, load_model
from load_data import get_data, get_data_val, get_data_extremes, get_data_extremes_val, get_toy_data, get_toy_data_val
from load_data import get_data_extremes_balanced, get_data_extremes_balanced_val
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, roc_auc_score, recall_score, accuracy_score
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from tensorflow.keras import backend as K
import sys
import warnings
from tensorflow.keras.callbacks import Callback


MAX_LEN = 300
BATCH_SIZE = 8
EPOCHS = 10
PATIENCE = 3
NUM_LABELS = 0
CLASSES = 0
PRE_TRAINED_MODEL_NAME = 'xlnet-base-cased'

XLNET_BASE = 0
XLNET_EXTR = 1
XLNET_PRETR_EXTR = 2
XLNET_EXTR_BAL = 3

PATH_RESULTS = './results/'

def set_seed(seed_value=11):
    # Set seed for reproducibility
    os.environ['PYTHONHASHSEED'] = str(0)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)


def aux_prec(y_true, y_pred):
    return precision_score(y_true, y_pred, average='macro', zero_division=0, labels=CLASSES)
def macro_precision(y_true, y_pred):
    preds = tf.math.argmax(y_pred, axis=1)
    labels = tf.math.argmax(y_true, axis=1)
    return tf.py_function(aux_prec, (labels, preds), tf.double)


def aux_rec(y_true, y_pred):
    return recall_score(y_true, y_pred, average='macro', zero_division=0, labels=CLASSES)
def macro_recall(y_true, y_pred):
    preds = tf.math.argmax(y_pred, axis=1)
    labels = tf.math.argmax(y_true, axis=1)
    return tf.py_function(aux_rec, (labels, preds), tf.double)


def aux_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro', labels=CLASSES, zero_division=0)
def macro_f1(y_true, y_pred):
    preds = tf.math.argmax(y_pred, axis=1)
    labels = tf.math.argmax(y_true, axis=1)
    return tf.py_function(aux_f1, (labels, preds), tf.double)


def aux_prec_micro(y_true, y_pred):
    return precision_score(y_true, y_pred, average='micro', zero_division=0, labels=CLASSES)
def micro_precision(y_true, y_pred):
    preds = tf.math.argmax(y_pred, axis=1)
    labels = tf.math.argmax(y_true, axis=1)
    return tf.py_function(aux_prec_micro, (labels, preds), tf.double)

def aux_rec_micro(y_true, y_pred):
    return recall_score(y_true, y_pred, average='micro', zero_division=0, labels=CLASSES)
def micro_recall(y_true, y_pred):
    preds = tf.math.argmax(y_pred, axis=1)
    labels = tf.math.argmax(y_true, axis=1)
    return tf.py_function(aux_rec_micro, (labels, preds), tf.double)

def aux_f1_micro(y_true, y_pred):
    return f1_score(y_true, y_pred, average='micro', labels=CLASSES, zero_division=0)
def micro_f1(y_true, y_pred):
    preds = tf.math.argmax(y_pred, axis=1)
    labels = tf.math.argmax(y_true, axis=1)
    return tf.py_function(aux_f1_micro, (labels, preds), tf.double)


def aux_prec_b(y_true, y_pred):
    return precision_score(y_true, y_pred, average='binary', zero_division=0, labels=CLASSES)
def b_precision(y_true, y_pred):
    preds = tf.compat.v1.to_int32(y_pred > 0.5)
    labels = y_true
    return tf.py_function(aux_prec_b, (labels, preds), tf.double)

def aux_rec_b(y_true, y_pred):
    return recall_score(y_true, y_pred, average='binary', zero_division=0, labels=CLASSES)
def b_recall(y_true, y_pred):
    preds = tf.compat.v1.to_int32(y_pred > 0.5)
    labels = y_true
    return tf.py_function(aux_rec_b, (labels, preds), tf.double)

def aux_f1_b(y_true, y_pred):
    return f1_score(y_true, y_pred, average='binary', labels=CLASSES, zero_division=0)
def b_f1(y_true, y_pred):
    preds = tf.compat.v1.to_int32(y_pred > 0.5)
    labels = y_true
    return tf.py_function(aux_f1_b, (labels, preds), tf.double)


def balanced_acc(y_true, y_pred):
    preds = tf.math.argmax(y_pred, axis=1)
    labels = tf.math.argmax(y_true, axis=1)
    return tf.py_function(balanced_accuracy_score, (labels, preds), tf.double)

def accuracy_bin(y_true, y_pred):
    preds = tf.compat.v1.to_int32(y_pred > 0.5)
    labels = y_true
    return tf.py_function(accuracy_score, (labels, preds), tf.double)


def aux_cust_f1(y_true, y_pred):
    prec_val = aux_prec(y_true,y_pred)
    rec_val = aux_rec(y_true, y_pred)
    return (2 * ((prec_val * rec_val) / (prec_val + rec_val + tf.keras.backend.epsilon())))
def sk_f1(y_true, y_pred):
    preds = tf.math.argmax(y_pred, axis=1)
    labels = tf.math.argmax(y_true, axis=1)
    return tf.py_function(aux_cust_f1, (preds, labels), tf.double)


def get_classes(num_classes):
    return np.arange(num_classes)


def set_class_weights(num_classes, train_set):
    class_weigths = {}
    if num_classes == 2:
        num_c = 2
        num_like = len([i for i, x in enumerate(train_set['rating']) if x == 1])
        num_dislike = len([i for i, x in enumerate(train_set['rating']) if x == 0])
        dislike_weigth = (num_dislike + num_like) / (num_c * num_dislike)
        like_weigth = (num_dislike + num_like) / (num_c * num_like)
        class_weigths = {
            0: dislike_weigth,
            1: like_weigth
        }
    elif num_classes == 5:
        num_1s = len([i for i, x in enumerate(train_set['rating']) if x == 0])
        num_2s = len([i for i, x in enumerate(train_set['rating']) if x == 1])
        num_3s = len([i for i, x in enumerate(train_set['rating']) if x == 2])
        num_4s = len([i for i, x in enumerate(train_set['rating']) if x == 3])
        num_5s = len([i for i, x in enumerate(train_set['rating']) if x == 4])
        num_examples = num_1s + num_2s + num_3s + num_4s + num_5s
        weigth_1s = num_examples / (num_classes * num_1s)
        weigth_2s = num_examples / (num_classes * num_2s)
        weigth_3s = num_examples / (num_classes * num_3s)
        weigth_4s = num_examples / (num_classes * num_4s)
        weigth_5s = num_examples / (num_classes * num_5s)
        class_weigths = {
            0: weigth_1s,
            1: weigth_2s,
            2: weigth_3s,
            3: weigth_4s,
            4: weigth_5s
        }
    return class_weigths


def pred_analysis_keras(model_name, preds, labels, run_time, path_conf, classes, performance):
    if NUM_LABELS != 2:
        preds = [np.argmax(x) for i, x in enumerate(preds)]
        labels = [np.argmax(x) for i, x in enumerate(labels)]
        flag_metrics = 'macro'
        acc_fun = balanced_accuracy_score
    else:
        flag_metrics = 'binary'
        acc_fun = accuracy_score
    performance['Model'].append(model_name)
    performance['Balanced Acc'].append(round(acc_fun(labels, preds), 4))
    performance['Precision'].append(round(precision_score(labels, preds, average=flag_metrics, zero_division=0, labels=CLASSES), 4))
    performance['Recall'].append(round(recall_score(labels, preds, average=flag_metrics, zero_division=0, labels=CLASSES), 4))
    performance['F1 Score'].append(round(f1_score(labels, preds, average=flag_metrics, zero_division=0, labels=CLASSES), 4))
    performance['Runtime'].append(run_time)
    df_results = pd.DataFrame(performance)
    final_path = path_conf + '_performance.csv'
    df_results.to_csv(final_path)
    matrix = confusion_matrix(labels, preds, labels=CLASSES)
    conf_matrix = pd.DataFrame(matrix, index=classes, columns=CLASSES)
    # Normalizing
    conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(15, 15))
    sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 15}, fmt='.2%')  # not normalized: ,fmt=''
    final_path = path_conf + '_conf_matrix.png'
    plt.savefig(final_path)
    return performance


def get_inputs_xlnet(reviews, tokenizer, max_len):
    inps = [tokenizer.encode_plus(t, max_length=max_len, pad_to_max_length=True, add_special_tokens=True,
                                  truncation=True) for t in reviews]
    inp_tok = np.array([a['input_ids'] for a in inps])
    type_ids = np.array([a['token_type_ids'] for a in inps])
    att_masks = np.array([a['attention_mask'] for a in inps])
    return inp_tok, type_ids, att_masks


def create_xlnet(mname, num_classes, fine_tuning=True):
    if num_classes == 2:
        last_fc = 1
        act_fun = 'sigmoid'
        loss_fun = 'binary_crossentropy'
        metrics = [accuracy_bin, b_precision, b_recall, b_f1]
    else:
        last_fc = num_classes
        loss_fun = 'categorical_crossentropy'
        act_fun = 'softmax'
        metrics = [balanced_acc, macro_precision, macro_recall, macro_f1]
    word_inputs = tf.keras.Input(shape=(MAX_LEN,), name='word_inputs', dtype='int32')
    id_inputs = tf.keras.Input(shape=(MAX_LEN,), name='id_inputs', dtype='int32')
    att_masks = tf.keras.Input(shape=(MAX_LEN,), name='mask_inputs', dtype='int32')
    xlnet = TFXLNetModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
    if not fine_tuning:
        xlnet.trainable = False
    xlnet_encodings = xlnet(input_ids=word_inputs, token_type_ids=id_inputs, attention_mask=att_masks)[0]
    doc_encoding = tf.squeeze(xlnet_encodings[:, -1:, :], axis=1)
    outputs = tf.keras.layers.Dense(last_fc, activation=act_fun, name='outputs')(doc_encoding)
    model = tf.keras.Model(inputs=[[word_inputs,id_inputs,att_masks]], outputs=[outputs])
    model.layers[3]._name = "last_hidden_state"
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5), loss=loss_fun,
                  metrics=metrics)
    return model


def create_xlnet_extr(mname, fine_tuning=True):
    metrics = [accuracy_bin, b_precision, b_recall, b_f1]
    word_inputs = tf.keras.Input(shape=(MAX_LEN,), name='word_inputs', dtype='int32')
    id_inputs = tf.keras.Input(shape=(MAX_LEN,), name='id_inputs', dtype='int32')
    att_masks = tf.keras.Input(shape=(MAX_LEN,), name='mask_inputs', dtype='int32')
    xlnet = TFXLNetModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
    if not fine_tuning:
        xlnet.trainable = False
    xlnet_encodings = xlnet(input_ids=word_inputs, token_type_ids=id_inputs, attention_mask=att_masks)[0]
    doc_encoding = tf.squeeze(xlnet_encodings[:, -1:, :], axis=1)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid', name='outputs')(doc_encoding)
    model = tf.keras.Model(inputs=[[word_inputs,id_inputs, att_masks]], outputs=[outputs])
    model.layers[3]._name = "last_hidden_state"
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5), loss='binary_crossentropy',
                  metrics=metrics)
    return model


def create_model(mname, num_classes, path_model, fine_tuning=False):
    if num_classes == 2:
        last_fc = 1
        act_fun = 'sigmoid'
        loss_fun = 'binary_crossentropy'
        metrics = [accuracy_bin, b_precision, b_recall, b_f1]
    else:
        last_fc = num_classes
        loss_fun = 'categorical_crossentropy'
        act_fun = 'softmax'
        metrics = [balanced_acc, macro_precision, macro_recall, macro_f1]
    loaded_model = create_xlnet_extr(mname, fine_tuning)
    loaded_model.load_weights(path_model)
    if not fine_tuning:
        loaded_model.trainable = False
    xlnet_encodings = loaded_model.get_layer('last_hidden_state').output [0]
    doc_encoding = tf.squeeze(xlnet_encodings[:, -1:, :], axis=1)
    outputs = tf.keras.layers.Dense(last_fc, activation=act_fun, name='outputs')(doc_encoding)
    model = tf.keras.Model(inputs=[loaded_model.input], outputs=[outputs])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5), loss=loss_fun,
                  metrics=metrics)
    return model


def train_model_es(model, x_train, y_train, x_val, y_val, class_weights, model_name, num_classes):
    if num_classes == 2:
        monitor = 'val_accuracy_bin'
    else:
        monitor = 'val_balanced_acc'
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=PATIENCE,
        min_delta=0.001,
        mode='max',
        restore_best_weights=True
    )
    init_time = time.time()
    history = model.fit(x=x_train, y=y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                        validation_data=(x_val, y_val), callbacks=[early_stopping],
                        class_weight=class_weights, verbose=2)
    end_time = time.time()
    run_time = round(end_time - init_time, 2)
    path_model = PATH_RESULTS + model_name + '/' + model_name + '_' + str(num_classes) + 'cl'
    model.save_weights(path_model)
    print('Model saved in path ', path_model)
    print('')
    if early_stopping.stopped_epoch == 0:
        num_epochs = EPOCHS
    else:
        num_epochs = early_stopping.stopped_epoch+1-PATIENCE
    return model, run_time, num_epochs


def train_model(model, x_train, y_train, class_weights, epochs, model_name, num_classes):
    init_time = time.time()
    history = model.fit(x=x_train, y=y_train, batch_size=BATCH_SIZE, epochs=epochs,
                        class_weight=class_weights, verbose=2)
    end_time = time.time()
    run_time = round(end_time - init_time, 2)
    path_model = PATH_RESULTS + model_name + '/' + model_name + '_' + str(num_classes) + 'cl'
    model.save_weights(path_model)
    print('Model saved in path ', path_model)
    print('')
    return model, run_time, path_model


def get_predictions_test(model, x_test, y_test, model_name, num_classes, run_time, df_results):
    preds = model.predict(x_test, batch_size=BATCH_SIZE)
    if num_classes == 2:
        preds = [1 if x > 0.5 else 0 for i, x in enumerate(preds)]
    else:
        preds = preds
    path_to_save_conf = PATH_RESULTS + model_name + '/' + model_name + str(num_classes) + 'cl'
    performance = pred_analysis_keras(model_name, preds, y_test, run_time, path_to_save_conf, CLASSES, df_results)
    return performance


def exps_base_es(df_results, fine_tuning=False, num_classes=5):
    model_name = 'XLNet_Base_ES'
    if fine_tuning:
        model_name = model_name + '_FT'
    model = create_xlnet(PRE_TRAINED_MODEL_NAME, num_classes, fine_tuning)
    xlnet_tokenizer = XLNetTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    train_set, val_set, test_set = get_data_val(num_classes, max_len=MAX_LEN)
    #train_set, val_set, test_set = get_toy_data_val(num_classes, max_len=MAX_LEN)
    class_weights = set_class_weights(num_classes, train_set)
    x_train, y_train = train_set['review_full'], train_set['rating']
    x_val, y_val = val_set['review_full'], val_set['rating']
    x_test, y_test = test_set['review_full'], test_set['rating']
    if num_classes != 2 :
        le = LabelEncoder()
        y_train_le = le.fit_transform(y_train)
        y_val_le = le.transform(y_val)
        y_test_le = le.transform(y_test)
        y_train_ = to_categorical(y_train_le)
        y_val_ = to_categorical(y_val_le)
        y_test_ = to_categorical(y_test_le)
    else:
        y_train_, y_val_, y_test_ = y_train, y_val, y_test
    print(model.summary())
    train_inp_tok, train_ids, train_segments = get_inputs_xlnet(x_train, xlnet_tokenizer, MAX_LEN)
    val_inp_tok, val_ids, val_segments = get_inputs_xlnet(x_val, xlnet_tokenizer, MAX_LEN)
    test_inp_tok, test_ids, test_segments = get_inputs_xlnet(x_test, xlnet_tokenizer, MAX_LEN)
    trained_model, run_time, final_epochs = train_model_es(model, [train_inp_tok, train_ids, train_segments], y_train_,
                                                           [val_inp_tok, val_ids, val_segments], y_val_,
                                                           class_weights, model_name, num_classes)
    performance = get_predictions_test(trained_model, [test_inp_tok, test_ids, test_segments], y_test_, model_name, num_classes, run_time, df_results)
    tf.keras.backend.clear_session()
    return final_epochs, performance


def exps_base(df_results, num_epochs, fine_tuning=False, num_classes=5):
    model_name = 'XLNet_Base'
    if fine_tuning:
        model_name = model_name + '_FT'
    model = create_xlnet(PRE_TRAINED_MODEL_NAME, num_classes, fine_tuning)
    xlnet_tokenizer = XLNetTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    train_set, test_set = get_data(num_classes, max_len=MAX_LEN)
    #train_set, test_set = get_toy_data(num_classes, max_len=MAX_LEN)
    class_weights = set_class_weights(num_classes, train_set)
    x_train, y_train = train_set['review_full'], train_set['rating']
    x_test, y_test = test_set['review_full'], test_set['rating']
    if num_classes != 2 :
        le = LabelEncoder()
        y_train_le = le.fit_transform(y_train)
        y_test_le = le.transform(y_test)
        y_train_ = to_categorical(y_train_le)
        y_test_ = to_categorical(y_test_le)
    else:
        y_train_, y_test_ = y_train, y_test
    print(model.summary())
    train_inp_tok, train_ids, train_segments = get_inputs_xlnet(x_train, xlnet_tokenizer, MAX_LEN)
    test_inp_tok, test_ids, test_segments = get_inputs_xlnet(x_test, xlnet_tokenizer, MAX_LEN)
    trained_model, run_time, path_model = train_model(model, [train_inp_tok, train_ids, train_segments], y_train_,
                                                      class_weights, num_epochs, model_name, num_classes)
    performance = get_predictions_test(trained_model, [test_inp_tok, test_ids, test_segments], y_test_, model_name,
                                       num_classes, run_time, df_results)
    tf.keras.backend.clear_session()
    return path_model, performance


def exps_extr_es(df_results, fine_tuning=False, num_classes=2, balanced=False):
    model_name = 'XLNet_Extr_ES'
    if fine_tuning:
        model_name = model_name + '_FT'
    model = create_xlnet_extr(PRE_TRAINED_MODEL_NAME, fine_tuning)
    xlnet_tokenizer = XLNetTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    if balanced:
        train_set, val_set, test_set = get_data_extremes_balanced_val(max_len=MAX_LEN)
        model_name = model_name + '_BAL'
    else:
        train_set, val_set, test_set = get_data_extremes_val(max_len=MAX_LEN)
    #train_set, val_set, test_set = get_toy_data_val(num_classes, max_len=MAX_LEN)
    class_weights = set_class_weights(num_classes, train_set)
    x_train, y_train = train_set['review_full'], train_set['rating']
    x_val, y_val = val_set['review_full'], val_set['rating']
    x_test, y_test = test_set['review_full'], test_set['rating']
    if num_classes != 2 :
        le = LabelEncoder()
        y_train_le = le.fit_transform(y_train)
        y_val_le = le.transform(y_val)
        y_test_le = le.transform(y_test)
        y_train_ = to_categorical(y_train_le)
        y_val_ = to_categorical(y_val_le)
        y_test_ = to_categorical(y_test_le)
    else:
        y_train_, y_val_, y_test_ = y_train, y_val, y_test
    print(model.summary())
    train_inp_tok, train_ids, train_segments = get_inputs_xlnet(x_train, xlnet_tokenizer, MAX_LEN)
    val_inp_tok, val_ids, val_segments = get_inputs_xlnet(x_val, xlnet_tokenizer, MAX_LEN)
    test_inp_tok, test_ids, test_segments = get_inputs_xlnet(x_test, xlnet_tokenizer, MAX_LEN)
    trained_model, run_time, final_epochs = train_model_es(model, [train_inp_tok, train_ids, train_segments], y_train_,
                                                           [val_inp_tok, val_ids, val_segments], y_val_,
                                                           class_weights, model_name, num_classes)
    performance = get_predictions_test(trained_model, [test_inp_tok, test_ids, test_segments], y_test_, model_name,
                                       num_classes, run_time, df_results)
    tf.keras.backend.clear_session()
    return final_epochs, performance


def exps_extr(df_results, num_epochs, fine_tuning=False, num_classes=2, balanced=False):
    model_name = 'XLNet_Extr'
    if fine_tuning:
        model_name = model_name + '_FT'
    if balanced:
        model_name = model_name + '_BAL'
    model = create_xlnet_extr(model_name, fine_tuning)
    xlnet_tokenizer = XLNetTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    #train_set, test_set = get_toy_data(num_classes, max_len=MAX_LEN)
    if balanced:
        train_set, test_set = get_data_extremes_balanced(max_len=MAX_LEN)
        model_name = model_name + '_BAL'
    else:
        train_set, test_set = get_data_extremes(max_len=MAX_LEN)
    class_weights = set_class_weights(num_classes, train_set)
    x_train, y_train = train_set['review_full'], train_set['rating']
    x_test, y_test = test_set['review_full'], test_set['rating']
    if num_classes != 2 :
        le = LabelEncoder()
        y_train_le = le.fit_transform(y_train)
        y_test_le = le.transform(y_test)
        y_train_ = to_categorical(y_train_le)
        y_test_ = to_categorical(y_test_le)
    else:
        y_train_, y_test_ = y_train, y_test
    print(model.summary())
    train_inp_tok, train_ids, train_segments = get_inputs_xlnet(x_train, xlnet_tokenizer, MAX_LEN)
    test_inp_tok, test_ids, test_segments = get_inputs_xlnet(x_test, xlnet_tokenizer, MAX_LEN)
    trained_model, run_time, path_model = train_model(model, [train_inp_tok, train_ids, train_segments], y_train_,
                                                      class_weights, num_epochs, model_name, num_classes)
    performance, get_predictions_test(trained_model, [test_inp_tok, test_ids, test_segments], y_test_, model_name,
                                      num_classes, run_time, df_results)
    tf.keras.backend.clear_session()
    return path_model, performance


def exps_pretrained_extr_es(df_results, path_model, num_classes=5, fine_tuning=False, balanced=False):
    model_name = 'XLNet_Pretr_Extr_ES'
    if fine_tuning:
        model_name = model_name + '_FT'
    if balanced:
        model_name = model_name + '_BAL'
    model = create_model(PRE_TRAINED_MODEL_NAME, num_classes, path_model, fine_tuning)
    xlnet_tokenizer = XLNetTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    #train_set, val_set, test_set = get_toy_data_val(num_classes, max_len=MAX_LEN)
    train_set, val_set, test_set = get_data_val(num_classes, max_len=MAX_LEN)
    class_weights = set_class_weights(num_classes, train_set)
    x_train, y_train = train_set['review_full'], train_set['rating']
    x_val, y_val = val_set['review_full'], val_set['rating']
    x_test, y_test = test_set['review_full'], test_set['rating']
    if num_classes != 2 :
        le = LabelEncoder()
        y_train_le = le.fit_transform(y_train)
        y_val_le = le.transform(y_val)
        y_test_le = le.transform(y_test)
        y_train_ = to_categorical(y_train_le)
        y_val_ = to_categorical(y_val_le)
        y_test_ = to_categorical(y_test_le)
    else:
        y_train_, y_val_, y_test_ = y_train, y_val, y_test
    print(model.summary())
    train_inp_tok, train_ids, train_segments = get_inputs_xlnet(x_train, xlnet_tokenizer, MAX_LEN)
    val_inp_tok, val_ids, val_segments = get_inputs_xlnet(x_val, xlnet_tokenizer, MAX_LEN)
    test_inp_tok, test_ids, test_segments = get_inputs_xlnet(x_test, xlnet_tokenizer, MAX_LEN)
    trained_model, run_time, final_epochs = train_model_es(model, [train_inp_tok, train_ids, train_segments], y_train_,
                                                           [val_inp_tok, val_ids, val_segments], y_val_,
                                                           class_weights, model_name, num_classes)
    performance = get_predictions_test(trained_model, [test_inp_tok, test_ids, test_segments], y_test_, model_name,
                                       num_classes, run_time, df_results)
    tf.keras.backend.clear_session()
    return final_epochs, performance

def exps_pretrained_extr(df_results, path_model, num_epochs, num_classes=5, fine_tuning=False, balanced=False):
    model_name = 'XLNet_Pretr_Extr'
    if fine_tuning:
        model_name = model_name + '_FT'
    if balanced:
        model_name = model_name + '_BAL'
    model = create_model(PRE_TRAINED_MODEL_NAME, num_classes, path_model, fine_tuning=fine_tuning)
    xlnet_tokenizer = XLNetTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    #train_set, test_set = get_toy_data(num_classes, max_len=MAX_LEN)
    train_set, test_set = get_data(num_classes, max_len=MAX_LEN)
    class_weights = set_class_weights(num_classes, train_set)
    x_train, y_train = train_set['review_full'], train_set['rating']
    x_test, y_test = test_set['review_full'], test_set['rating']
    if num_classes != 2 :
        le = LabelEncoder()
        y_train_le = le.fit_transform(y_train)
        y_test_le = le.transform(y_test)
        y_train_ = to_categorical(y_train_le)
        y_test_ = to_categorical(y_test_le)
    else:
        y_train_, y_test_ = y_train, y_test
    print(model.summary())
    train_inp_tok, train_ids, train_segments = get_inputs_xlnet(x_train, xlnet_tokenizer, MAX_LEN)
    test_inp_tok, test_ids, test_segments = get_inputs_xlnet(x_test, xlnet_tokenizer, MAX_LEN)
    trained_model, run_time, path_model = train_model(model, [train_inp_tok, train_ids, train_segments], y_train_,
                                                      class_weights, num_epochs, model_name, num_classes)
    performance = get_predictions_test(trained_model, [test_inp_tok, test_ids, test_segments], y_test_, model_name,
                                       num_classes, run_time, df_results)
    tf.keras.backend.clear_session()
    return path_model, performance



def set_experiment(exp_type, df_results=None, early_s=False, fine_tuning=False, num_classes=5, path_model=None, num_epochs=None, balanced=False):
    final_epochs = -1
    path_trained_model = ''
    if early_s:
        if exp_type == XLNET_BASE:
            final_epochs, performance = exps_base_es(df_results, fine_tuning, num_classes=num_classes)
        elif exp_type == XLNET_EXTR:
            final_epochs, performance = exps_extr_es(df_results, fine_tuning, num_classes=2, balanced=False)
        elif exp_type == XLNET_PRETR_EXTR:
            final_epochs, performance = exps_pretrained_extr_es(df_results, path_model, num_classes, fine_tuning, balanced=balanced)
        else: # XLNET_EXTR_BAL
            final_epochs, performance = exps_extr_es(df_results, fine_tuning, num_classes=2, balanced=True)
    else:
        if exp_type == XLNET_BASE:
            path_trained_model, performance = exps_base(df_results, num_epochs, fine_tuning, num_classes=num_classes)
        elif exp_type == XLNET_EXTR:
            path_trained_model, performance = exps_extr(df_results, num_epochs, fine_tuning, num_classes=2, balanced=False)
        elif exp_type == XLNET_PRETR_EXTR:
            path_trained_model, performance = exps_pretrained_extr(df_results, path_model, num_epochs, num_classes, fine_tuning, balanced=balanced)
        else: # XLNET_EXTR_BAL
            path_trained_model, performance = exps_extr(df_results, num_epochs, fine_tuning, num_classes=2, balanced=True)

    return final_epochs, path_trained_model, performance


if not sys.warnoptions:
    warnings.simplefilter("ignore")

set_seed()


gpu = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

performance = {'Model': [],
               'Balanced Acc': [],
               'Precision': [],
               'Recall': [],
               'F1 Score': [],
               'Runtime': []}

#EXPS 5 CLASES

exp = XLNET_BASE
num_cl = 5
CLASSES = get_classes(num_cl)
NUM_LABELS = num_cl


print('Training XLNet base sin FT... ')
final_epochs, _, performance = set_experiment(exp_type=exp, early_s=True, num_classes=num_cl, fine_tuning=False, df_results=performance)
_, model_path, performance = set_experiment(exp_type=exp, early_s=False, num_classes=num_cl, num_epochs=final_epochs, fine_tuning=False, df_results=performance)
print('Model path = ', model_path)
print('')


print('Training XLNet base con FT... ')
final_epochs, _, performance = set_experiment(exp_type=exp, early_s=True, num_classes=num_cl, fine_tuning=True, df_results=performance)
_, model_path, performance = set_experiment(exp_type=exp, early_s=False, num_classes=num_cl, num_epochs=final_epochs, fine_tuning=True, df_results=performance)
print('Model path = ', model_path)
print('')


exp = XLNET_EXTR
num_cl = 2
CLASSES = get_classes(num_cl)
NUM_LABELS = num_cl
print('Training XLNet Extremes con FT... ')
final_epochs, _, performance = set_experiment(exp_type=exp, early_s=True, num_classes=num_cl, fine_tuning=True, df_results=performance)
_, model_path, performance = set_experiment(exp_type=exp, early_s=False, num_classes=num_cl, num_epochs=final_epochs, fine_tuning=True, df_results=performance)
print('Model path = ', model_path)
print('')


exp = XLNET_PRETR_EXTR
num_cl = 5 #2
CLASSES = get_classes(num_cl)
NUM_LABELS = num_cl
print('Training with XLNet Extremes in new model without FT... ')
final_epochs, _, performance = set_experiment(exp_type=exp, early_s=True, num_classes=num_cl, fine_tuning=False, path_model=model_path, df_results=performance, balanced=False)
_, model_path_final, performance = set_experiment(exp_type=exp, early_s=False, num_classes=num_cl, num_epochs=final_epochs, fine_tuning=False, path_model=model_path, df_results=performance, balanced=False)
print('Model path = ', model_path_final)
print('')

exp = XLNET_PRETR_EXTR
num_cl = 5 #2
CLASSES = get_classes(num_cl)
NUM_LABELS = num_cl
print('Training with XLNet Extremes in new model with FT... ')
final_epochs, _, performance = set_experiment(exp_type=exp, early_s=True, num_classes=num_cl, fine_tuning=True, path_model=model_path, df_results=performance, balanced=False)
_, model_path_final, performance = set_experiment(exp_type=exp, early_s=False, num_classes=num_cl, num_epochs=final_epochs, fine_tuning=True, path_model=model_path, df_results=performance, balanced=False)
print('Model path = ', model_path_final)
print('')


exp = XLNET_EXTR_BAL
num_cl = 2
CLASSES = get_classes(num_cl)
NUM_LABELS = num_cl
print('Training XLNet Extremes Balanced con FT... ')
final_epochs, _, performance = set_experiment(exp_type=exp, early_s=True, num_classes=num_cl, fine_tuning=True, df_results=performance, balanced=True)
_, model_path, performance = set_experiment(exp_type=exp, early_s=False, num_classes=num_cl, num_epochs=final_epochs, fine_tuning=True, df_results=performance, balanced=True)
print('Model path = ', model_path)
print('')


exp = XLNET_PRETR_EXTR
num_cl = 5
CLASSES = get_classes(num_cl)
NUM_LABELS = num_cl
print('Training with XLNet Extremes Balanced in new model without FT... ')
final_epochs, _, performance = set_experiment(exp_type=exp, early_s=True, num_classes=num_cl, fine_tuning=False, path_model=model_path, df_results=performance, balanced=True)
_, model_path_final, performance = set_experiment(exp_type=exp, early_s=False, num_classes=num_cl, num_epochs=final_epochs, fine_tuning=False, path_model=model_path, df_results=performance, balanced=True)
print('Model path = ', model_path_final)
print('')

exp = XLNET_PRETR_EXTR
num_cl = 5
CLASSES = get_classes(num_cl)
NUM_LABELS = num_cl
print('Training with XLNet Extremes Balanced in new model with FT... ')
final_epochs, _, performance = set_experiment(exp_type=exp, early_s=True, num_classes=num_cl, fine_tuning=True, path_model=model_path, df_results=performance, balanced=True)
_, model_path_final, performance = set_experiment(exp_type=exp, early_s=False, num_classes=num_cl, num_epochs=final_epochs, fine_tuning=True, path_model=model_path, df_results=performance, balanced=True)
print('Model path = ', model_path_final)
print('')


df_results = pd.DataFrame(performance)
# df_results = pd.DataFrame(pd.Series(performance))
print(df_results.head())
path_to_save_perf = PATH_RESULTS + 'xlnet_exps_performace.csv'
print('Performance saved at : ', path_to_save_perf)
df_results.to_csv(path_to_save_perf)

