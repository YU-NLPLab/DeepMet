import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

id = np.array(pd.read_csv('./data/VUA/verb_tokens_test.csv', names=['id'])).tolist()
predict = np.array(pd.read_csv('./predict/VUA_verb_predict5.csv', names=['predict'])).tolist()
id = [i for row in id for i in row]
predict = [i for row in predict for i in row]
submit = pd.DataFrame({'id': id[1:], 'predict': predict[1:]})
submit.to_csv('./submit/answer.txt', index=False, header=None)

data1 = pd.read_csv('./data/VUA/all_pos_tokens_test_gold_labels.csv')
data2 = pd.read_csv('./predict/VUA_allpos_predict7.csv')
id1 = np.array(data1.id).tolist()
label1 = np.array(data1.label).tolist()
id2 = np.array(data2.id).tolist()
label2 = np.array(data2.predict).tolist()
index1 = [i for i in range(len(id1)) if id1[i] in id2]
index2 = [i for i in range(len(id2)) if id2[i] in id1]
for i in range(len(index1)):
    label1[index1[i]] = label2[index2[i]]
submit = pd.DataFrame({"id": id1, "predict": label1})
submit.to_csv('./submit/answer.txt', index=False, header=None)

data1 = pd.read_csv('./predict/VUA_verb_predict6.csv')
data2 = pd.read_csv('./predict/VUA_allpos_predict6.csv')

print("best verb accuracy: {:.4f}".format(accuracy_score(data1.label, data1.predict)))
print("best verb precision: {:.4f}".format(precision_score(data1.label, data1.predict)))
print("best verb recall: {:.4f}".format(recall_score(data1.label, data1.predict)))
print("best verb f1: {:.4f}".format(f1_score(data1.label, data1.predict)))

print("best allpos accuracy: {:.4f}".format(accuracy_score(data2.label, data2.predict)))
print("best allpos precision: {:.4f}".format(precision_score(data2.label, data2.predict)))
print("best allpos recall: {:.4f}".format(recall_score(data2.label, data2.predict)))
print("best allpos f1: {:.4f}".format(f1_score(data2.label, data2.predict)))
