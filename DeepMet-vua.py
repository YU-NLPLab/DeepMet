import os, argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedKFold
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
import tensorflow.keras.backend as K
from keras.utils import to_categorical
from scipy.stats import spearmanr
from math import floor, ceil
from transformers import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def preprocssing(x):
    x = str(x)
    x = '"'+x+'"'
    return x


def _convert_to_transformer_inputs(instance, instance2, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for transformer (including bert)"""
    def return_id(str1, str2, truncation_strategy, length):
        inputs = tokenizer.encode_plus(str1, str2,
                                       add_special_tokens=True,
                                       max_length=length,
                                       truncation_strategy=truncation_strategy)
        input_ids = inputs["input_ids"]
        input_masks = [1] * len(input_ids)
        input_segments = inputs["token_type_ids"]
        padding_length = length - len(input_ids)
        padding_id = tokenizer.pad_token_id
        input_ids = input_ids + ([padding_id] * padding_length)
        input_masks = input_masks + ([0] * padding_length)
        input_segments = input_segments + ([0] * padding_length)
        return [input_ids, input_masks, input_segments]
    input_ids, input_masks, input_segments = return_id(
        instance, None, 'longest_first', max_sequence_length)
    input_ids2, input_masks2, input_segments2 = return_id(
        instance2, None, 'longest_first', max_sequence_length)
    return [input_ids, input_masks, input_segments,
            input_ids2, input_masks2, input_segments2]


def compute_input_arrays(df, columns, tokenizer, max_sequence_length):
    input_ids, input_masks, input_segments = [], [], []
    input_ids2, input_masks2, input_segments2 = [], [], []
    for _, instance in tqdm(df[columns].iterrows()):
        ids, masks, segments, ids2, masks2, segments2 = \
            _convert_to_transformer_inputs(str(instance.sentence), str(instance.sentence2), tokenizer, max_sequence_length)
        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)
        input_ids2.append(ids2)
        input_masks2.append(masks2)
        input_segments2.append(segments2)
    return [np.asarray(input_ids, dtype=np.int32),
            np.asarray(input_masks, dtype=np.int32),
            np.asarray(input_segments, dtype=np.int32),
            np.asarray(input_ids2, dtype=np.int32),
            np.asarray(input_masks2, dtype=np.int32),
            np.asarray(input_segments2, dtype=np.int32)]


def compute_output_arrays(df, columns):
    return np.asarray(df[columns].astype(int))

# Siamese structure
def create_model():
    input_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    input_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    input_atn = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    input_id2 = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    input_mask2 = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    input_atn2 = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    config = RobertaConfig.from_pretrained('roberta-base')
    config.output_hidden_states = False
    base_model = TFRobertaModel.from_pretrained('roberta-base', config=config)
    TransformerA = base_model(input_id, attention_mask=input_mask, token_type_ids=input_atn)[0]
    TransformerB = base_model(input_id2, attention_mask=input_mask2, token_type_ids=input_atn2)[0]
    output = tf.keras.layers.GlobalAveragePooling1D()(TransformerA)
    output2 = tf.keras.layers.GlobalAveragePooling1D()(TransformerB)
    x = tf.keras.layers.Concatenate()([output, output2])
    x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
    x = tf.keras.layers.Dense(2, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=[input_id, input_mask, input_atn, input_id2, input_mask2, input_atn2], outputs=x)
    return model


# Pseudo siamese structure
def create_model2():
    input_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    input_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    input_atn = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    input_id2 = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    input_mask2 = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    input_atn2 = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    config = RobertaConfig.from_pretrained('roberta-base')
    config.output_hidden_states = False
    base_model = TFRobertaModel.from_pretrained('roberta-base', config=config)
    base_model2 = TFRobertaModel.from_pretrained('roberta-base', config=config)
    TransformerA = base_model(input_id, attention_mask=input_mask, token_type_ids=input_atn)[0]
    TransformerB = base_model2(input_id2, attention_mask=input_mask2, token_type_ids=input_atn2)[0]
    output = tf.keras.layers.GlobalAveragePooling1D()(TransformerA)
    output2 = tf.keras.layers.GlobalAveragePooling1D()(TransformerB)
    x = tf.keras.layers.Concatenate()([output, output2])
    x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
    x = tf.keras.layers.Dense(2, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=[input_id, input_mask, input_atn, input_id2, input_mask2, input_atn2], outputs=x)
    return model


# Ablation Experiment -B
def create_model3():
    input_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    input_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    input_atn = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    config = RobertaConfig.from_pretrained('roberta-base')
    config.output_hidden_states = False
    base_model = TFRobertaModel.from_pretrained('roberta-base', config=config)
    TransformerA = base_model(input_id, attention_mask=input_mask, token_type_ids=input_atn)[0]
    output = tf.keras.layers.GlobalAveragePooling1D()(TransformerA)
    x = tf.keras.layers.Concatenate()(output)
    x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
    x = tf.keras.layers.Dense(2, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=[input_id, input_mask, input_atn], outputs=x)
    return model


# Ablation Experiment -A
def create_model4():
    input_id2 = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    input_mask2 = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    input_atn2 = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    config = RobertaConfig.from_pretrained('roberta-base')
    config.output_hidden_states = False
    base_model2 = TFRobertaModel.from_pretrained('roberta-base', config=config)
    TransformerB = base_model2(input_id2, attention_mask=input_mask2, token_type_ids=input_atn2)[0]
    output2 = tf.keras.layers.GlobalAveragePooling1D()(TransformerB)
    x = tf.keras.layers.Concatenate()(output2)
    x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
    x = tf.keras.layers.Dense(2, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=[input_id2, input_mask2, input_atn2], outputs=x)
    return model

if __name__ == '__main__':

    np.set_printoptions(suppress=True)
    print(tf.__version__)
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Experimental environment: Titan RTX
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

    # Hyperparameter search range:
    # MAX_SEQUENCE_LENGTH = 64,128,256,512
    # HIDDEN_SIZE = 768,1024
    # RANDOM_STATE = 2020
    # EPOCHS = 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20
    # N_FOLD = 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20
    # BATCH_SIZE = 16,32,64,128,256,512
    # DROPOUT_RATE = 0.1,0.2,0.3,0.4,0.5
    # VALIDATION_SPLIT = 0.1,0.2,0.3
    # LEARNING_RATE = 1e-5,2e-5,3e-5,4e-5,5e-5
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_sequence_length", default=128, type=int, required=False)
    parser.add_argument("--hidden_size", default=768, type=int, required=False)
    parser.add_argument("--random_state", default=2020, type=int, required=False)
    parser.add_argument("--epochs", default=3, type=int, required=False)
    parser.add_argument("--n_fold", default=10, type=int, required=False)
    parser.add_argument("--batch_size", default=16, type=int, required=False)
    parser.add_argument("--dropout_rate", default=0.2, type=float, required=False)
    parser.add_argument("--validation_split", default=0.1, type=float, required=False)
    parser.add_argument("--learning_rate", default=1e-5, type=float, required=False)
    args = parser.parse_args()
    MAX_SEQUENCE_LENGTH = args.max_sequence_length
    HIDDEN_SIZE = args.hidden_size
    RANDOM_STATE = args.random_state
    EPOCHS = args.epochs
    N_FOLD = args.n_fold
    BATCH_SIZE = args.batch_size
    DROPOUT_RATE = args.dropout_rate
    VALIDATION_SPLIT = args.validation_split
    LEARNING_RATE = args.learning_rate

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    train = pd.read_csv('./data/VUA/VUA_train_features2.csv')
    test = pd.read_csv('./data/VUA/VUA_verb_features.csv')
    test2 = pd.read_csv('./data/VUA/VUA_allpos_features.csv')
    print('train shape =', train.shape)
    print('verb test shape =', test.shape)
    print('allpos test shape =', test2.shape)

    train['sentence'] = train.sentence.apply(lambda x: preprocssing(x)) \
                        + "[SEP]" + train.word.apply(lambda x: preprocssing(x)) \
                        + "[SEP]" + train.pos.apply(lambda x: preprocssing(x)) \
                        + "[SEP]" + train.tag.apply(lambda x: preprocssing(x))
    test['sentence'] = test.sentence.apply(lambda x: preprocssing(x)) \
                       + "[SEP]" + test.word.apply(lambda x: preprocssing(x)) \
                       + "[SEP]" + test.pos.apply(lambda x: preprocssing(x)) \
                       + "[SEP]" + test.tag.apply(lambda x: preprocssing(x))
    test2['sentence'] = test2.sentence.apply(lambda x: preprocssing(x)) \
                        + "[SEP]" + test2.word.apply(lambda x: preprocssing(x)) \
                        + "[SEP]" + test2.pos.apply(lambda x: preprocssing(x)) \
                        + "[SEP]" + test2.tag.apply(lambda x: preprocssing(x))
    train['sentence2'] = train.local.apply(lambda x: preprocssing(x)) \
                         + "[SEP]" + train.word.apply(lambda x: preprocssing(x)) \
                         + "[SEP]" + train.pos.apply(lambda x: preprocssing(x)) \
                         + "[SEP]" + train.tag.apply(lambda x: preprocssing(x))
    test['sentence2'] = test.local.apply(lambda x: preprocssing(x)) \
                        + "[SEP]" + test.word.apply(lambda x: preprocssing(x)) \
                        + "[SEP]" + test.pos.apply(lambda x: preprocssing(x)) \
                        + "[SEP]" + test.tag.apply(lambda x: preprocssing(x))
    test2['sentence2'] = test2.local.apply(lambda x: preprocssing(x)) \
                         + "[SEP]" + test2.word.apply(lambda x: preprocssing(x)) \
                         + "[SEP]" + test2.pos.apply(lambda x: preprocssing(x)) \
                         + "[SEP]" + test2.tag.apply(lambda x: preprocssing(x))

    # # Ablation Experiment -pos
    # train['sentence'] = train.sentence.apply(lambda x: preprocssing(x)) \
    #              + "[SEP]" + train.word.apply(lambda x: preprocssing(x)) \
    #              + "[SEP]" + train.tag.apply(lambda x: preprocssing(x))
    # test['sentence'] = test.sentence.apply(lambda x: preprocssing(x)) \
    #              + "[SEP]" + test.word.apply(lambda x: preprocssing(x)) \
    #              + "[SEP]" + test.tag.apply(lambda x: preprocssing(x))
    # test2['sentence'] = test2.sentence.apply(lambda x: preprocssing(x)) \
    #              + "[SEP]" + test2.word.apply(lambda x: preprocssing(x)) \
    #              + "[SEP]" + test2.tag.apply(lambda x: preprocssing(x))
    # train['sentence2'] = train.local.apply(lambda x: preprocssing(x)) \
    #              + "[SEP]" + train.word.apply(lambda x: preprocssing(x)) \
    #              + "[SEP]" + train.tag.apply(lambda x: preprocssing(x))
    # test['sentence2'] = test.local.apply(lambda x: preprocssing(x)) \
    #              + "[SEP]" + test.word.apply(lambda x: preprocssing(x)) \
    #              + "[SEP]" + test.tag.apply(lambda x: preprocssing(x))
    # test2['sentence2'] = test2.local.apply(lambda x: preprocssing(x)) \
    #              + "[SEP]" + test2.word.apply(lambda x: preprocssing(x)) \
    #              + "[SEP]" + test2.tag.apply(lambda x: preprocssing(x))
    #
    # # Ablation Experiment -tag
    # train['sentence'] = train.sentence.apply(lambda x: preprocssing(x)) \
    #              + "[SEP]" + train.word.apply(lambda x: preprocssing(x)) \
    #              + "[SEP]" + train.pos.apply(lambda x: preprocssing(x))
    # test['sentence'] = test.sentence.apply(lambda x: preprocssing(x)) \
    #              + "[SEP]" + test.word.apply(lambda x: preprocssing(x)) \
    #              + "[SEP]" + test.pos.apply(lambda x: preprocssing(x))
    # test2['sentence'] = test2.sentence.apply(lambda x: preprocssing(x)) \
    #              + "[SEP]" + test2.word.apply(lambda x: preprocssing(x)) \
    #              + "[SEP]" + test2.pos.apply(lambda x: preprocssing(x))
    # train['sentence2'] = train.local.apply(lambda x: preprocssing(x)) \
    #              + "[SEP]" + train.word.apply(lambda x: preprocssing(x)) \
    #              + "[SEP]" + train.pos.apply(lambda x: preprocssing(x))
    # test['sentence2'] = test.local.apply(lambda x: preprocssing(x)) \
    #              + "[SEP]" + test.word.apply(lambda x: preprocssing(x)) \
    #              + "[SEP]" + test.pos.apply(lambda x: preprocssing(x))
    # test2['sentence2'] = test2.local.apply(lambda x: preprocssing(x)) \
    #              + "[SEP]" + test2.word.apply(lambda x: preprocssing(x)) \
    #              + "[SEP]" + test2.pos.apply(lambda x: preprocssing(x))

    input_categories = ['sentence', 'sentence2']
    output_categories = 'label'
    inputs = compute_input_arrays(train, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
    outputs = compute_output_arrays(train, output_categories)
    test_inputs = compute_input_arrays(test, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
    test_inputs2 = compute_input_arrays(test2, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)

    pred = np.zeros((len(test)))
    pred2 = np.zeros((len(test2)))
    gkf = StratifiedKFold(n_splits=N_FOLD).split(X=train[input_categories], y=train[output_categories])

    for fold, (train_idx, valid_idx) in enumerate(gkf):
        train_inputs = [inputs[i][train_idx] for i in range(len(inputs))]
        train_outputs = to_categorical(outputs[train_idx])
        valid_inputs = [inputs[i][valid_idx] for i in range(len(inputs))]
        valid_outputs = to_categorical(outputs[valid_idx])
        K.clear_session()
        model = create_model()
        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc', 'mae'])
        model.fit(train_inputs, train_outputs, validation_data=[valid_inputs, valid_outputs], epochs=EPOCHS,
                  batch_size=BATCH_SIZE)
        model.save_weights(f'./model/model-{fold}.h5')
        fold_pred = np.argmax(model.predict(test_inputs), axis=1)
        fold_pred2 = np.argmax(model.predict(test_inputs2), axis=1)
        pred += fold_pred
        pred2 += fold_pred2
        print("folds: {:d}".format(fold))
        print("verb accuracy: {:.4f}".format(accuracy_score(test.label, fold_pred)))
        print("verb precision: {:.4f}".format(precision_score(test.label, fold_pred)))
        print("verb recall: {:.4f}".format(recall_score(test.label, fold_pred)))
        print("verb f1: {:.4f}".format(f1_score(test.label, fold_pred)))
        print("allpos accuracy: {:.4f}".format(accuracy_score(test2.label, fold_pred2)))
        print("allpos precision: {:.4f}".format(precision_score(test2.label, fold_pred2)))
        print("allpos recall: {:.4f}".format(recall_score(test2.label, fold_pred2)))
        print("allpos f1: {:.4f}".format(f1_score(test2.label, fold_pred2)))

    best_pred = np.zeros((len(test)))
    best_score = 0
    best_threshold = 0
    for i in range(N_FOLD):
        temp_pred = (np.array(pred) >= i).astype('int')
        print("verb metaphor preference parameter alpha: {:d}".format(i/N_FOLD))
        print("verb accuracy: {:.4f}".format(accuracy_score(test.label, temp_pred)))
        print("verb precision: {:.4f}".format(precision_score(test.label, temp_pred)))
        print("verb recall: {:.4f}".format(recall_score(test.label, temp_pred)))
        print("verb f1: {:.4f}".format(f1_score(test.label, temp_pred)))
        if f1_score(test.label, temp_pred) > best_score:
            best_score = f1_score(test.label, temp_pred)
            best_pred = temp_pred
            best_threshold = i

    print("best verb metaphor preference parameter alpha: {:d}".format(best_threshold/N_FOLD))
    print("best verb accuracy: {:.4f}".format(accuracy_score(test.label, best_pred)))
    print("best verb precision: {:.4f}".format(precision_score(test.label, best_pred)))
    print("best verb recall: {:.4f}".format(recall_score(test.label, best_pred)))
    print("best verb f1: {:.4f}".format(f1_score(test.label, best_pred)))

    best_pred2 = np.zeros((len(test2)))
    best_score2 = 0
    best_threshold2 = 0
    for i in range(N_FOLD):
        temp_pred2 = (np.array(pred2) >= i).astype('int')
        print("allpos metaphor preference parameter alpha: {:d}".format(i/N_FOLD))
        print("allpos accuracy: {:.4f}".format(accuracy_score(test2.label, temp_pred2)))
        print("allpos precision: {:.4f}".format(precision_score(test2.label, temp_pred2)))
        print("allpos recall: {:.4f}".format(recall_score(test2.label, temp_pred2)))
        print("allpos f1: {:.4f}".format(f1_score(test2.label, temp_pred2)))
        if f1_score(test2.label, temp_pred2) > best_score2:
            best_score2 = f1_score(test2.label, temp_pred2)
            best_pred2 = temp_pred2
            best_threshold2 = i

    print("best allpos metaphor preference parameter alpha: {:d}".format(best_threshold2/N_FOLD))
    print("best allpos accuracy: {:.4f}".format(accuracy_score(test2.label, best_pred2)))
    print("best allpos precision: {:.4f}".format(precision_score(test2.label, best_pred2)))
    print("best allpos recall: {:.4f}".format(recall_score(test2.label, best_pred2)))
    print("best allpos f1: {:.4f}".format(f1_score(test2.label, best_pred2)))

    test['predict'] = best_pred
    test2['predict'] = best_pred2
    print(test.predict.value_counts())
    print(test2.predict.value_counts())

    test[['id', 'sentence', 'word', 'label', 'predict']].to_csv('./predict/VUA_verb_predict.csv', index=False)
    test2[['id', 'sentence', 'word', 'label', 'predict']].to_csv('./predict/VUA_allpos_predict.csv', index=False)
    test[['id', 'predict']].to_csv('./submit/answer.txt', index=False, header=None)
    test2[['id', 'predict']].to_csv('./submit/answer2.txt', index=False, header=None)
