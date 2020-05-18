import re
import numpy as np
import pandas as pd
from os.path import join
from os import walk

id = []
sentence = []
word = []
label = []
offset2token ={}
texts_dir = './data/TOEFI/train'
for (dirpath, dirnames, filenames) in walk(texts_dir):
	for f in filenames:
		txt_id = f.split('.')[0]
		with open (join(dirpath,f), 'r') as f:
			sent_id = 1
			for line in f:
				line = re.sub('\n', '', line)
				tokens = line.strip().split()
				offset_id = 1
				for t in tokens:
					offset2token['_'.join((txt_id,str(sent_id),str(offset_id)))] = t
					id.extend(['_'.join((txt_id,str(sent_id),str(offset_id)))])
					sentence.extend([re.sub('M_', '', line)])
					word.extend([re.sub('M_', '', t)])
					label.extend([1 if "M_" in t else 0])
					offset_id += 1
				sent_id += 1

data = pd.DataFrame({"id": id, "sentence": sentence, "word": word, "label": label})
data.to_csv('./data/TOEFI/TOEFI_train.csv', index=False)

id = []
sentence = []
word = []
label = []
offset2token ={}
texts_dir = './data/TOEFI/test'
for (dirpath, dirnames, filenames) in walk(texts_dir):
	for f in filenames:
		txt_id = f.split('.')[0]
		with open (join(dirpath,f), 'r') as f:
			sent_id = 1
			for line in f:
				line = re.sub('\n', '', line)
				tokens = line.strip().split()
				offset_id = 1
				for t in tokens:
					offset2token['_'.join((txt_id,str(sent_id),str(offset_id)))] = t
					id.extend(['_'.join((txt_id,str(sent_id),str(offset_id)))])
					sentence.extend([re.sub('M_', '', line)])
					word.extend([re.sub('M_', '', t)])
					label.extend([1 if "M_" in t else 0])
					offset_id += 1
				sent_id += 1

data = pd.DataFrame({"id": id, "sentence": sentence, "word": word, "label": label})
data.to_csv('./data/TOEFI/TOEFI_test_all.csv', index=False)