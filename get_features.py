import re, nltk
import pandas as pd
import spacy
from spacy import displacy
from nltk.chunk import conlltags2tree
from nltk.corpus import wordnet as wn
from nltk.corpus import verbnet as vn
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from vaderSentiment import vaderSentiment
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


RANDOM_STATE = 2020
VALIDATION_SPLIT = 0.1
nlp = spacy.load('en_core_web_lg')
analyzer = vaderSentiment.SentimentIntensityAnalyzer()


def label(x):
    return 1 if "1" in x else 0


def get_lemma(x):
    return ' '.join([token.lemma_ for token in nlp(x)])


def get_pos(x):
    return ' '.join([token.pos_ for token in nlp(x)])


def get_word_pos(x):
    try:
        return str(nlp(x)[0].pos_)
    except:
        return "NONE"


def get_tag(x):
        return ' '.join([token.tag_ for token in nlp(x)])


def get_word_tag(x):
    try:
        return str(nlp(x)[0].tag_)
    except:
        return "NONE"


def get_dep(x):
    return ' '.join([token.dep_ for token in nlp(x)])


def get_word_dep(x):
    return str(nlp(x)[0].dep_)


def get_entity(x):
    return ' '.join([str(ent) for ent in nlp(x).ents])


def join(x):
    return ' '.join(x)


def get_entity_tag(x):
    return [(token.text, token.tag_, "{0}-{1}".format(token.ent_iob_, token.ent_type_) if token.ent_iob_ != 'O' else token.ent_iob_) for token in nlp(x)]


def get_chunk(x):
    return [(chunk.text, chunk.label_, chunk.root.text) for chunk in nlp(x).noun_chunks]


def get_knowledge(x):
    try:
        if x == None or x == ' ':
            return
        elif nlp(x)[0].pos_ == 'NOUN':
            return wn.synset(nlp(x)[0].lemma_ + '.n.01').definition()
        elif nlp(x)[0].pos_ == 'VERB':
            return wn.synset(nlp(x)[0].lemma_ + '.v.01').definition()
        elif nlp(x)[0].pos_ == 'ADJ':
            return wn.synset(nlp(x)[0].lemma_ + '.a.01').definition()
        else:
            return
    except:
        return


def get_knowledges(x):
    return ' '.join([str(get_knowledge(i)) for i in x.split(' ') if get_knowledge(i)!=None])


def get_verb_knowledge(x):
    try:
        return wn.synset(nlp(x)[0].lemma_ + '.v.01').definition()
    except:
        return "none"


def get_sentiment(x):
    result = analyzer.polarity_scores(x)
    if result['pos'] >= result['neg'] and result['pos'] >= result['neu']:
        return "postive"
    elif result['neg'] >= result['pos'] and result['neg'] >= result['neu']:
        return "negative"
    else:
        return "neutral"


def get_sentiment_score(x):
    return analyzer.polarity_scores(x)


def set_none(x):
    return re.sub('none', '', str(x))


data = pd.read_csv('./data/VUA/VUA_formatted_train.csv', encoding='latin-1')
data["lemma"] = data.sentence.apply(lambda x: get_lemma(x))
data["pos"] = data.sentence.apply(lambda x: get_pos(x))
data["tag"] = data.sentence.apply(lambda x: get_tag(x))
data["dep"] = data.sentence.apply(lambda x: get_dep(x))
data["entity"] = data.sentence.apply(lambda x: get_entity(x))
data["entity_tag"] = data.sentence.apply(lambda x: get_entity_tag(x))
data["chunk"] = data.sentence.apply(lambda x: get_chunk(x))
data["knowledge"] = data.sentence.apply(lambda x: get_verb_knowledge(x))
data["sentiment"] = data.sentence.apply(lambda x: get_sentiment(x))
data["sentiment_score"] = data.sentence.apply(lambda x: get_sentiment_score(x))
data.to_csv('./data/VUA/VUA_linguistic_features_train.csv', index=False, header=True)

data = pd.read_csv('./data/VUA/VUA_formatted_val.csv', encoding='latin-1')
data["lemma"] = data.sentence.apply(lambda x: get_lemma(x))
data["pos"] = data.sentence.apply(lambda x: get_pos(x))
data["tag"] = data.sentence.apply(lambda x: get_tag(x))
data["dep"] = data.sentence.apply(lambda x: get_dep(x))
data["entity"] = data.sentence.apply(lambda x: get_entity(x))
data["entity_tag"] = data.sentence.apply(lambda x: get_entity_tag(x))
data["chunk"] = data.sentence.apply(lambda x: get_chunk(x))
data["knowledge"] = data.sentence.apply(lambda x: get_verb_knowledge(x))
data["sentiment"] = data.sentence.apply(lambda x: get_sentiment(x))
data["sentiment_score"] = data.sentence.apply(lambda x: get_sentiment_score(x))
data.to_csv('./data/VUA/VUA_linguistic_features_val.csv', index=False, header=True)

data = pd.read_csv('./data/VUA/VUA_formatted_test.csv', encoding='latin-1')
data["lemma"] = data.sentence.apply(lambda x: get_lemma(x))
data["pos"] = data.sentence.apply(lambda x: get_pos(x))
data["tag"] = data.sentence.apply(lambda x: get_tag(x))
data["dep"] = data.sentence.apply(lambda x: get_dep(x))
data["entity"] = data.sentence.apply(lambda x: get_entity(x))
data["entity_tag"] = data.sentence.apply(lambda x: get_entity_tag(x))
data["chunk"] = data.sentence.apply(lambda x: get_chunk(x))
data["knowledge"] = data.sentence.apply(lambda x: get_verb_knowledge(x))
data["sentiment"] = data.sentence.apply(lambda x: get_sentiment(x))
data["sentiment_score"] = data.sentence.apply(lambda x: get_sentiment_score(x))
data.to_csv('./data/VUA/VUA_linguistic_features_test.csv', index=False, header=True)

data = pd.read_csv('./data/MOH-X/MOH-X_train.csv', encoding='latin-1')
data["lemma"] = data.sentence.apply(lambda x: get_lemma(x))
data["pos"] = data.sentence.apply(lambda x: get_pos(x))
data["tag"] = data.sentence.apply(lambda x: get_tag(x))
data["dep"] = data.sentence.apply(lambda x: get_dep(x))
data["entity"] = data.sentence.apply(lambda x: get_entity(x))
data["entity_tag"] = data.sentence.apply(lambda x: get_entity_tag(x))
data["chunk"] = data.sentence.apply(lambda x: get_chunk(x))
data["knowledge"] = data.sentence.apply(lambda x: get_verb_knowledge(x))
data["sentiment"] = data.sentence.apply(lambda x: get_sentiment(x))
data["sentiment_score"] = data.sentence.apply(lambda x: get_sentiment_score(x))
data.to_csv('./data/MOH-X/MOH-X_linguistic_features_train.csv', index=False, header=True)

data = pd.read_csv('./data/MOH-X/MOH-X_test.csv', encoding='latin-1')
data["lemma"] = data.sentence.apply(lambda x: get_lemma(x))
data["pos"] = data.sentence.apply(lambda x: get_pos(x))
data["tag"] = data.sentence.apply(lambda x: get_tag(x))
data["dep"] = data.sentence.apply(lambda x: get_dep(x))
data["entity"] = data.sentence.apply(lambda x: get_entity(x))
data["entity_tag"] = data.sentence.apply(lambda x: get_entity_tag(x))
data["chunk"] = data.sentence.apply(lambda x: get_chunk(x))
data["knowledge"] = data.sentence.apply(lambda x: get_verb_knowledge(x))
data["sentiment"] = data.sentence.apply(lambda x: get_sentiment(x))
data["sentiment_score"] = data.sentence.apply(lambda x: get_sentiment_score(x))
data.to_csv('./data/MOH-X/MOH-X_linguistic_features_test.csv', index=False, header=True)

data = pd.read_csv('./data/TroFi/TroFi_train.csv', encoding='latin-1')
data["lemma"] = data.sentence.apply(lambda x: get_lemma(x))
data["pos"] = data.sentence.apply(lambda x: get_pos(x))
data["tag"] = data.sentence.apply(lambda x: get_tag(x))
data["dep"] = data.sentence.apply(lambda x: get_dep(x))
data["entity"] = data.sentence.apply(lambda x: get_entity(x))
data["entity_tag"] = data.sentence.apply(lambda x: get_entity_tag(x))
data["chunk"] = data.sentence.apply(lambda x: get_chunk(x))
data["knowledge"] = data.sentence.apply(lambda x: get_verb_knowledge(x))
data["sentiment"] = data.sentence.apply(lambda x: get_sentiment(x))
data["sentiment_score"] = data.sentence.apply(lambda x: get_sentiment_score(x))
data.to_csv('./data/TroFi/TroFi_linguistic_features_train.csv', index=False, header=True)

data = pd.read_csv('./data/TroFi/TroFi_test.csv', encoding='latin-1')
data["lemma"] = data.sentence.apply(lambda x: get_lemma(x))
data["pos"] = data.sentence.apply(lambda x: get_pos(x))
data["tag"] = data.sentence.apply(lambda x: get_tag(x))
data["dep"] = data.sentence.apply(lambda x: get_dep(x))
data["entity"] = data.sentence.apply(lambda x: get_entity(x))
data["entity_tag"] = data.sentence.apply(lambda x: get_entity_tag(x))
data["chunk"] = data.sentence.apply(lambda x: get_chunk(x))
data["knowledge"] = data.sentence.apply(lambda x: get_verb_knowledge(x))
data["sentiment"] = data.sentence.apply(lambda x: get_sentiment(x))
data["sentiment_score"] = data.sentence.apply(lambda x: get_sentiment_score(x))
data.to_csv('./data/TroFi/TroFi_linguistic_features_test.csv', index=False, header=True)

train = pd.read_csv('./data/VUA_feature.csv')
train, test = train_test_split(train, test_size=VALIDATION_SPLIT, random_state=RANDOM_STATE)
train, val = train_test_split(train, test_size=VALIDATION_SPLIT, random_state=RANDOM_STATE)
train.to_csv('./data/VUA_feature_train.csv', index=False, header=True)
val.to_csv('./data/VUA_feature_val.csv', index=False, header=True)
test.to_csv('./data/VUA_feature_test.csv', index=False, header=True)

train = pd.read_csv('./data/TroFi.csv')
train, test = train_test_split(train, test_size=VALIDATION_SPLIT, random_state=RANDOM_STATE)
train, val = train_test_split(train, test_size=VALIDATION_SPLIT, random_state=RANDOM_STATE)
train.to_csv('./data/TroFi_train.csv', index=False, header=True)
val.to_csv('./data/TroFi_val.csv', index=False, header=True)
test.to_csv('./data/TroFi_test.csv', index=False, header=True)

train = pd.read_csv('./data/MOH.csv')
train, test = train_test_split(train, test_size=VALIDATION_SPLIT, random_state=RANDOM_STATE)
train, val = train_test_split(train, test_size=VALIDATION_SPLIT, random_state=RANDOM_STATE)
train.to_csv('./data/MOH_train.csv', index=False, header=True)
val.to_csv('./data/MOH_val.csv', index=False, header=True)
test.to_csv('./data/MOH_test.csv', index=False, header=True)

train = pd.read_csv('./data/TroFi-X.csv')
train, test = train_test_split(train, test_size=VALIDATION_SPLIT, random_state=RANDOM_STATE)
train, val = train_test_split(train, test_size=VALIDATION_SPLIT, random_state=RANDOM_STATE)
train.to_csv('./data/TroFi-X_train.csv', index=False, header=True)
val.to_csv('./data/TroFi-X_val.csv', index=False, header=True)
test.to_csv('./data/TroFi-X_test.csv', index=False, header=True)

train = pd.read_csv('./data/TroFi/TroFi_formatted_all3737.csv')
train, test = train_test_split(train, test_size=VALIDATION_SPLIT, random_state=RANDOM_STATE)
train.to_csv('./data/TroFi/TroFi_train.csv', index=False, header=True)
test.to_csv('./data/TroFi/TroFi_test.csv', index=False, header=True)

data = pd.read_csv('./data/VUA/VUA_verb_test.csv')
data["pos"] = data.word.apply(lambda x: get_word_pos(x))
data["tag"] = data.word.apply(lambda x: get_word_tag(x))
data.to_csv('./data/VUA2/VUA_verb_features.csv', index=False, header=True)
print("verb")

data = pd.read_csv('./data/VUA/VUA_allpos_test.csv')
data["pos"] = data.word.apply(lambda x: get_word_pos(x))
data["tag"] = data.word.apply(lambda x: get_word_tag(x))
data.to_csv('./data/VUA2/VUA_allpos_features.csv', index=False, header=True)
print("allpos")

data = pd.read_csv('./data/VUA/VUA_train.csv')
data["pos"] = data.word.apply(lambda x: get_word_pos(x))
data["tag"] = data.word.apply(lambda x: get_word_tag(x))
data.to_csv('./data/VUA2/VUA_train_features.csv', index=False, header=True)

data = pd.read_csv('./data/TOEFI/TOEFI_verb_test.csv')
data["pos"] = data.word.apply(lambda x: get_word_pos(x))
data["tag"] = data.word.apply(lambda x: get_word_tag(x))
data.to_csv('./data/TOEFI/TOEFI_verb_features.csv', index=False, header=True)
print("verb")

data = pd.read_csv('./data/TOEFI/TOEFI_allpos_test.csv')
data["pos"] = data.word.apply(lambda x: get_word_pos(x))
data["tag"] = data.word.apply(lambda x: get_word_tag(x))
data.to_csv('./data/TOEFI/TOEFI_allpos_features.csv', index=False, header=True)
print("allpos")

data = pd.read_csv('./data/TOEFI/TOEFI_train.csv')
data["pos"] = data.word.apply(lambda x: get_word_pos(x))
data["tag"] = data.word.apply(lambda x: get_word_tag(x))
data.to_csv('./data/TOEFI/TOEFI_train_features.csv', index=False, header=True)

data = pd.read_csv('./data/TOEFI/TOEFI_verb_features.csv')
print(data['sentence'].str.count(' ').mean())
print(data['sentence'].str.count(' ').max())
print(data['local'].str.count(' ').mean())
print(data['local'].str.count(' ').max())

data = pd.read_csv('./data/TOEFI/TOEFI_allpos_features.csv')
print(data['sentence'].str.count(' ').mean())
print(data['sentence'].str.count(' ').max())
print(data['local'].str.count(' ').mean())
print(data['local'].str.count(' ').max())


data = pd.read_csv('./data/TOEFI/TOEFI_train_features2.csv')
print(data['sentence'].str.count(' ').mean())
print(data['sentence'].str.count(' ').max())
print(data['local'].str.count(' ').mean())
print(data['local'].str.count(' ').max())


data = pd.read_csv('./data/VUA/VUA_verb_features.csv')
print(data['sentence'].str.count(' ').mean())
print(data['sentence'].str.count(' ').max())
print(data['local'].str.count(' ').mean())
print(data['local'].str.count(' ').max())


data = pd.read_csv('./data/VUA/VUA_allpos_features.csv')
print(data['sentence'].str.count(' ').mean())
print(data['sentence'].str.count(' ').max())
print(data['local'].str.count(' ').mean())
print(data['local'].str.count(' ').max())


data = pd.read_csv('./data/VUA/VUA_train_features2.csv')
print(data['sentence'].str.count(' ').mean())
print(data['sentence'].str.count(' ').max())
print(data['local'].str.count(' ').mean())
print(data['local'].str.count(' ').max())
