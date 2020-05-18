import re, string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords

sw = stopwords.words("english")
def remove_stopwords(x):
    x = str(x)
    return ' '.join([w for w in x.split(' ') if w not in sw])

puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '+', '\\', '•',  '~', '@', '£',
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…',
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─',
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞',
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

def clean_text(x):
    x = str(x)
    x = re.sub('.*<SEP>', '', x)
    x = re.sub('<ENT>', '', x)
    x = re.sub('[0-9]*', '', x)
    for punct in puncts:
        x = x.replace(punct, '')
    return x

def clean_numbers(x):
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x

def shuffle(d):
    len_ = len(d)
    times = 2
    for i in range(times):
        index = np.random.choice(len_, 2)
        d[index[0]],d[index[1]] = d[index[1]],d[index[0]]
    return d

def dropout(d, p=0.1):
    len_ = len(d)
    index = np.random.choice(len_, int(len_ * p))
    for i in index:
        d[i] = ' '
    return d

def clean(x):
    x = x.split(' ')
    return x

def data_augment(train_df, n):
    train_text = list(train_df["text"].values)
    train_target = list(train_df["target"].values)
    l = len(train_text)
    for i in range(l):
        for j in range(n):
            item = clean(train_text[i])
            d1 = shuffle(item)
            d11 = ' '.join(d1)
            train_text.extend([d11])
            train_target.append(train_target[i])
            d2 = dropout(item)
            d22 = ' '.join(d2)
            train_text.extend([d22])
            train_target.append(train_target[i])
    augment_df = pd.DataFrame({"text":train_text,"target":train_target})
    augment_df = augment_df.sample(frac=1)
    return augment_df

mispell_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have",
                "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not",
                "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will",
                "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am",
                "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have",
                "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have",
                "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",
                "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have",
                "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not",
                "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have",
                "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would",
                "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
                "so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is",
                "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is",
                "they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have",
                "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would",
                "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have",
                "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have",
                "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will",
                "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have",
                "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have",
                "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would",
                "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are",
                "you've": "you have", 'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling',
                'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor',
                'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ',
                'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do',
                'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do',
                'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation',
                'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum',
                'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota',
                'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp',
                'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization'}

def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re

mispellings, mispellings_re = _get_mispell(mispell_dict)

def replace_typical_misspell(text):
    def replace(match):
        return mispellings[match.group(0)]
    return mispellings_re.sub(replace, text)

train = pd.read_csv('./data/VUAsequence/VUA_seq_formatted_train.csv', encoding='latin-1')
train_fragid = np.array(train.txt_id).tolist()
train_senid = np.array(train.sen_ix).tolist()
train_sentences = np.array(train.sentence).tolist()
train_lables = np.array(train.label_seq).tolist()
train_genres = np.array(train.genre).tolist()

train_id = []
train_text = []
train_words = []
train_lable = []
train_genre = []

for i, labels in enumerate(train_lables):
    labels = re.sub('\[', '', labels)
    labels = re.sub('\]', '', labels)
    labels = re.sub(',', '', labels)
    tokens = labels.split()
    train_words.extend(train_sentences[i].split())
    for j in range(len(tokens)):
        train_text.extend([train_sentences[i]])
        train_id.extend([train_fragid[i] + '_' + train_senid[i] + '_' + str(j+1)])
        train_genre.extend([train_genres[i]])
        if tokens[j] == "0":
            train_lable.append(0)
        else:
            train_lable.append(1)

data = pd.DataFrame({"id": train_id, "sentence": train_text, "word": train_words, "label": train_lable, "genre": train_genre})
data.to_csv('./data/VUAsequence/VUA_SEQ_train.csv', index=False)

test = pd.read_csv('./data/VUAsequence/VUA_seq_formatted_test.csv', encoding='latin-1')
test_fragid = np.array(test.txt_id).tolist()
test_senid = np.array(test.sen_ix).tolist()
test_sentences = np.array(test.sentence).tolist()
test_lables = np.array(test.label_seq).tolist()
test_genres = np.array(test.genre).tolist()

test_id = []
test_text = []
test_words = []
test_lable = []
test_genre = []

for i, labels in enumerate(test_lables):
    labels = re.sub('\[', '', labels)
    labels = re.sub('\]', '', labels)
    labels = re.sub(',', '', labels)
    tokens = labels.split()
    test_words.extend(test_sentences[i].split())
    for j in range(len(tokens)):
        test_text.extend([test_sentences[i]])
        test_id.extend([str(test_fragid[i]) + '_' + str(test_senid[i]) + '_' + str(j+1)])
        test_genre.extend([test_genres[i]])
        if tokens[j] == "0":
            test_lable.append(0)
        else:
            test_lable.append(1)

data = pd.DataFrame({"id": test_id, "sentence": test_text, "word": test_words, "label": test_lable, "genre": test_genre})
data.to_csv('./data/VUAsequence/VUA_SEQ_test.csv', index=False)

data1 = pd.read_csv('./data/VUAsequence/all_pos_tokens_test_gold_labels.csv')
data2 = pd.read_csv('./data/VUAsequence/VUA_SEQ_test_all.csv')
id = np.array(data2.id).tolist()
sentences = np.array(data2.sentence).tolist()
word = np.array(data2.word).tolist()
label = np.array(data2.label).tolist()
genre = np.array(data2.genre).tolist()
id2 = np.array(data1.id).tolist()
index = []
for i in range(len(id2)):
    for j in range(len(id)):
        if id2[i] == id[j]:
            index.extend([j])

idlist = []
sentenceslist = []
wordlist = []
labellist = []
genrelist = []
for i in range(len(index)):
    idlist.append(id[index[i]])
    sentenceslist.append(sentences[index[i]])
    wordlist.append(word[index[i]])
    labellist.append(label[index[i]])
    genrelist.append(genre[index[i]])

data = pd.DataFrame({"id": idlist, "sentence": sentenceslist, "word": wordlist, "label": labellist, "genre": genrelist})
data.to_csv('./data/VUAsequence/VUA_SEQ_test2.csv', index=False)

data1 = pd.read_csv('./data/VUAsequence/all_pos_tokens_train_gold_labels.csv')
data2 = pd.read_csv('./data/VUAsequence/VUA_SEQ_train_all.csv')
id = np.array(data2.id).tolist()
sentences = np.array(data2.sentence).tolist()
word = np.array(data2.word).tolist()
label = np.array(data2.label).tolist()
genre = np.array(data2.genre).tolist()
id2 = np.array(data1.id).tolist()
index = [i for i in range(len(id2)) if id2[i] in id]

idlist = []
sentenceslist = []
wordlist = []
labellist = []
genrelist = []
for i in range(len(index)):
    idlist.append(id[index[i]])
    sentenceslist.append(sentences[index[i]])
    wordlist.append(word[index[i]])
    labellist.append(label[index[i]])
    genrelist.append(genre[index[i]])

data = pd.DataFrame({"id": idlist, "sentence": sentenceslist, "word": wordlist, "label": labellist, "genre": genrelist})
data.to_csv('./data/VUAsequence/VUA_SEQ_train.csv', index=False)

data1 = pd.read_csv('./data/TOEFI/toefl_verb_test_tokens.csv')
data2 = pd.read_csv('./data/TOEFI/TOEFI_test_all.csv')
id = np.array(data2.id).tolist()
sentences = np.array(data2.sentence).tolist()
word = np.array(data2.word).tolist()
label = np.array(data2.label).tolist()
id2 = np.array(data1.id).tolist()
for i in range(len(id2)):
    id2[i] = id2[i].strip().split("_")[0]+'_'+id2[i].strip().split("_")[1]+'_'+id2[i].strip().split("_")[2]
index = []
for i in range(len(id2)):
    for j in range(len(id)):
        if id2[i] == id[j]:
            index.extend([j])
idlist = []
sentenceslist = []
wordlist = []
labellist = []
genrelist = []
for i in range(len(index)):
    idlist.append(id[index[i]])
    sentenceslist.append(sentences[index[i]])
    wordlist.append(word[index[i]])
    labellist.append(label[index[i]])

data = pd.DataFrame({"id": idlist, "sentence": sentenceslist, "word": wordlist, "label": labellist})
data.to_csv('./data/TOEFI/TOEFI_verb_test.csv', index=False)

data1 = pd.read_csv('./data/TOEFI/toefl_all_pos_test_tokens.csv')
data2 = pd.read_csv('./data/TOEFI/TOEFI_test_all.csv')
id = np.array(data2.id).tolist()
sentences = np.array(data2.sentence).tolist()
word = np.array(data2.word).tolist()
label = np.array(data2.label).tolist()
id2 = np.array(data1.id).tolist()
for i in range(len(id2)):
    id2[i] = id2[i].strip().split("_")[0]+'_'+id2[i].strip().split("_")[1]+'_'+id2[i].strip().split("_")[2]
index = []
for i in range(len(id2)):
    for j in range(len(id)):
        if id2[i] == id[j]:
            index.extend([j])
idlist = []
sentenceslist = []
wordlist = []
labellist = []
genrelist = []
for i in range(len(index)):
    idlist.append(id[index[i]])
    sentenceslist.append(sentences[index[i]])
    wordlist.append(word[index[i]])
    labellist.append(label[index[i]])

data = pd.DataFrame({"id": idlist, "sentence": sentenceslist, "word": wordlist, "label": labellist})
data.to_csv('./data/TOEFI/TOEFI_allpos_test.csv', index=False)

data1 = pd.read_csv('./data/VUA/verb_tokens_test_gold_labels.csv')
data2 = pd.read_csv('./data/VUA/VUA_test_all.csv')
id = np.array(data2.id).tolist()
sentences = np.array(data2.sentence).tolist()
word = np.array(data2.word).tolist()
label = np.array(data2.label).tolist()
id2 = np.array(data1.id).tolist()
index = []
for i in range(len(id2)):
    for j in range(len(id)):
        if id2[i] == id[j]:
            index.extend([j])
idlist = []
sentenceslist = []
wordlist = []
labellist = []
genrelist = []
for i in range(len(index)):
    idlist.append(id[index[i]])
    sentenceslist.append(sentences[index[i]])
    wordlist.append(word[index[i]])
    labellist.append(label[index[i]])

data = pd.DataFrame({"id": idlist, "sentence": sentenceslist, "word": wordlist, "label": labellist})
data.to_csv('./data/VUA/VUA_verb_test.csv', index=False)

data1 = pd.read_csv('./data/VUA/all_pos_tokens_test_gold_labels.csv')
data2 = pd.read_csv('./data/VUA/VUA_test_all.csv')
id = np.array(data2.id).tolist()
sentences = np.array(data2.sentence).tolist()
word = np.array(data2.word).tolist()
label = np.array(data2.label).tolist()
id2 = np.array(data1.id).tolist()
index = []
for i in range(len(id2)):
    for j in range(len(id)):
        if id2[i] == id[j]:
            index.extend([j])
idlist = []
sentenceslist = []
wordlist = []
labellist = []
genrelist = []
for i in range(len(index)):
    idlist.append(id[index[i]])
    sentenceslist.append(sentences[index[i]])
    wordlist.append(word[index[i]])
    labellist.append(label[index[i]])

data = pd.DataFrame({"id": idlist, "sentence": sentenceslist, "word": wordlist, "label": labellist})
data.to_csv('./data/VUA/VUA_allpos_test.csv', index=False)

data = pd.read_csv('./data/VUA/VUA_verb_features.csv')
id = np.array(data.id).tolist()
sentence = np.array(data.sentence).tolist()
word = np.array(data.word).tolist()
label = np.array(data.label).tolist()
pos = np.array(data.pos).tolist()
tag = np.array(data.tag).tolist()

local = []
for i in range(len(sentence)):
    flag = 0
    w = word[i]
    slice = sentence[i].split(',')
    for j in range(len(slice)):
        if w in slice[j] and flag == 0:
            local.extend([slice[j]])
            flag = 1

data = pd.DataFrame({"id": id, "sentence": sentence, "word": word, "label": label, "pos": pos, "tag": tag, "local": local})
data.to_csv('./data/VUA/VUA_verb_features.csv', index=False)

data = pd.read_csv('./data/VUA/VUA_allpos_features.csv')
id = np.array(data.id).tolist()
sentence = np.array(data.sentence).tolist()
word = np.array(data.word).tolist()
label = np.array(data.label).tolist()
pos = np.array(data.pos).tolist()
tag = np.array(data.tag).tolist()

local = []
for i in range(len(sentence)):
    flag = 0
    w = word[i]
    slice = sentence[i].split(',')
    for j in range(len(slice)):
        if w in slice[j] and flag == 0:
            local.extend([slice[j]])
            flag = 1

data = pd.DataFrame({"id": id, "sentence": sentence, "word": word, "label": label, "pos": pos, "tag": tag, "local": local})
data.to_csv('./data/VUA/VUA_allpos_features.csv', index=False)

data = pd.read_csv('./data/TOEFI/TOEFI_verb_features.csv')
id = np.array(data.id).tolist()
sentence = np.array(data.sentence).tolist()
word = np.array(data.word).tolist()
label = np.array(data.label).tolist()
pos = np.array(data.pos).tolist()
tag = np.array(data.tag).tolist()

local = []
for i in range(len(sentence)):
    flag = 0
    w = word[i]
    slice = sentence[i].split(',')
    for j in range(len(slice)):
        if w in slice[j] and flag == 0:
            local.extend([slice[j]])
            flag = 1

data = pd.DataFrame({"id": id, "sentence": sentence, "word": word, "label": label, "pos": pos, "tag": tag, "local": local})
data.to_csv('./data/TOEFI/TOEFI_verb_features.csv', index=False)

data = pd.read_csv('./data/TOEFI/TOEFI_allpos_features.csv')
id = np.array(data.id).tolist()
sentence = np.array(data.sentence).tolist()
word = np.array(data.word).tolist()
label = np.array(data.label).tolist()
pos = np.array(data.pos).tolist()
tag = np.array(data.tag).tolist()

local = []
for i in range(len(sentence)):
    flag = 0
    w = word[i]
    slice = sentence[i].split(',')
    for j in range(len(slice)):
        if w in slice[j] and flag == 0:
            local.extend([slice[j]])
            flag = 1

data = pd.DataFrame({"id": id, "sentence": sentence, "word": word, "label": label, "pos": pos, "tag": tag, "local": local})
data.to_csv('./data/TOEFI/TOEFI_allpos_features.csv', index=False)

data = pd.read_csv('./data/TOEFI/TOEFI_train_features.csv')
id = np.array(data.id).tolist()
sentence = np.array(data.sentence).tolist()
word = np.array(data.word).tolist()
label = np.array(data.label).tolist()
pos = np.array(data.pos).tolist()
tag = np.array(data.tag).tolist()
id2 = np.array(data.id).tolist()

index = []
punc = string.punctuation
for i in range(len(sentence)):
    w = word[i]
    if w in punc:
        index.extend([i])

id = [id[i] for i in range(len(id2)) if i not in index]
sentence = [sentence[i] for i in range(len(id2)) if i not in index]
word = [word[i] for i in range(len(id2)) if i not in index]
label = [label[i] for i in range(len(id2)) if i not in index]
pos = [pos[i] for i in range(len(id2)) if i not in index]
tag= [tag[i] for i in range(len(id2)) if i not in index]

local = []
for i in range(len(sentence)):
    flag = 0
    w = word[i]
    slice = sentence[i].split(',')
    for j in range(len(slice)):
        if w in slice[j] and flag == 0:
            local.extend([slice[j]])
            flag = 1

data = pd.DataFrame({"id": id, "sentence": sentence, "word": word, "label": label, "pos": pos, "tag": tag, "local": local})
data.to_csv('./data/TOEFI/TOEFI_train_features.csv', index=False)

data = pd.read_csv('./data/VUA/VUA_train_features.csv')
id2 = np.array(data.id).tolist()
sentence2 = np.array(data.sentence).tolist()
word2 = np.array(data.word).tolist()
label2 = np.array(data.label).tolist()
pos2 = np.array(data.pos).tolist()
tag2 = np.array(data.tag).tolist()

id = []
sentence = []
word = []
label = []
pos = []
tag = []
index = []
punc = string.punctuation
for i in range(len(sentence2)):
    w = str(word2[i])
    p = str(pos2[i])
    if w in punc or ',' in w or 'nan' in w or p == 'PUNCT' or p == 'DET' or p == 'ADP' or p == 'CCONJ':
        index.extend([i])

for i in range(len(id2)):
    if i not in index:
        id.extend([id2[i]])
        sentence.extend([sentence2[i]])
        word.extend([word2[i]])
        label.extend([label2[i]])
        pos.extend([pos2[i]])
        tag.extend([tag2[i]])

local = []
others = []
for i in range(len(sentence)):
    flag = 0
    w = str(word[i])
    slice = sentence[i].split(',')
    for j in range(len(slice)):
        if w in str(slice[j]) and flag == 0:
            local.extend([str(slice[j])])
            flag = 1
        elif j == len(slice)-1 and flag == 0:
            others.extend([w + '||' + str(slice[j])])

data = pd.DataFrame({"id": id, "sentence": sentence, "word": word, "label": label, "pos": pos, "tag": tag, "local": local})
data.to_csv('./data/VUA/VUA_train_features2.csv', index=False)

data = pd.read_csv('./data/TOEFI/TOEFI_train_features.csv')
id2 = np.array(data.id).tolist()
sentence2 = np.array(data.sentence).tolist()
word2 = np.array(data.word).tolist()
label2 = np.array(data.label).tolist()
pos2 = np.array(data.pos).tolist()
tag2 = np.array(data.tag).tolist()

id = []
sentence = []
word = []
label = []
pos = []
tag = []
index = []
punc = string.punctuation
for i in range(len(sentence2)):
    w = str(word2[i])
    p = str(pos2[i])
    if w in punc or ',' in w or 'nan' in w:
        index.extend([i])

for i in range(len(id2)):
    if i not in index:
        id.extend([id2[i]])
        sentence.extend([sentence2[i]])
        word.extend([word2[i]])
        label.extend([label2[i]])
        pos.extend([pos2[i]])
        tag.extend([tag2[i]])

local = []
others = []
for i in range(len(sentence)):
    flag = 0
    w = str(word[i])
    slice = sentence[i].split(',')
    for j in range(len(slice)):
        if w in str(slice[j]) and flag == 0:
            local.extend([str(slice[j])])
            flag = 1
        elif j == len(slice)-1 and flag == 0:
            others.extend([w + '||' + str(slice[j])])

data = pd.DataFrame({"id": id, "sentence": sentence, "word": word, "label": label, "pos": pos, "tag": tag, "local": local})
data.to_csv('./data/TOEFI/TOEFI_train_features2.csv', index=False)

data = pd.read_csv('./data/VUA/VUA_verb_features.csv')
print(data['pos'].value_counts())
print(data['tag'].value_counts())

data2 = pd.read_csv('./data/VUA/VUA_allpos_features.csv')
print(data2['pos'].value_counts())
print(data2['tag'].value_counts())

data3 = pd.read_csv('./data/VUA/VUA_train_features.csv')
print(data3['pos'].value_counts())
print(data3['tag'].value_counts())
