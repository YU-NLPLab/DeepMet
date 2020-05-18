import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data = pd.read_csv('./predict/VUA_verb_predict.csv')
data2 = pd.read_csv('./predict/VUA_verb_predict2.csv')
data3 = pd.read_csv('./predict/VUA_bert_predict3.csv')
data4 = pd.read_csv('./predict/VUA_verb_predict4.csv')
data5 = pd.read_csv('./predict/VUA_verb_predict5.csv')
data6 = pd.read_csv('./predict/VUA_verb_predict6.csv')
data7 = pd.read_csv('./predict/VUA_verb_predict7.csv')
data8 = pd.read_csv('./predict/VUA_verb_predict8.csv')
data9 = pd.read_csv('./predict/VUA_verb_predict9.csv')
data10 = pd.read_csv('./predict/VUA_verb_predict10.csv')
data11 = pd.read_csv('./predict/VUA_verb_predict11.csv')

pred = (data.predict + data2.predict + data3.predict + data4.predict + data5.predict + data6.predict + data7.predict + data8.predict + data9.predict + data10.predict + data11.predict).tolist()
print(len(pred))
print(pred)

best_pred = np.zeros((len(pred)))
best_score = 0
best_threshold = 0
for i in range(8):
    temp_pred = (np.array(pred) >= i).astype('int')
    print("verb threshold: {:d}".format(i))
    print("verb accuracy: {:.4f}".format(accuracy_score(data.label, temp_pred)))
    print("verb precision: {:.4f}".format(precision_score(data.label, temp_pred)))
    print("verb recall: {:.4f}".format(recall_score(data.label, temp_pred)))
    print("verb f1: {:.4f}".format(f1_score(data.label, temp_pred)))
    if f1_score(data.label, temp_pred) > best_score:
        best_score = f1_score(data.label, temp_pred)
        best_pred = temp_pred
        best_threshold = i

print("best verb threshold: {:d}".format(best_threshold))
print("best verb accuracy: {:.4f}".format(accuracy_score(data.label, best_pred)))
print("best verb precision: {:.4f}".format(precision_score(data.label, best_pred)))
print("best verb recall: {:.4f}".format(recall_score(data.label, best_pred)))
print("best verb f1: {:.4f}".format(f1_score(data.label, best_pred)))

data['predict'] = best_pred
print(data.predict.value_counts())
data[['id', 'predict']].to_csv('./submit/answer.txt', index=False, header=None)

data = pd.read_csv('./predict/VUA_allpos_predict.csv')
data2 = pd.read_csv('./predict/VUA_allpos_predict2.csv')
data3 = pd.read_csv('./predict/VUA_allpos_predict3.csv')
data4 = pd.read_csv('./predict/VUA_allpos_predict4.csv')
data5 = pd.read_csv('./predict/VUA_allpos_predict5.csv')
data6 = pd.read_csv('./predict/VUA_allpos_predict6.csv')
data7 = pd.read_csv('./predict/VUA_allpos_predict7.csv')
data8 = pd.read_csv('./predict/VUA_allpos_predict8.csv')
data9 = pd.read_csv('./predict/VUA_allpos_predict9.csv')
data10 = pd.read_csv('./predict/VUA_allpos_predict10.csv')

pred = (data.predict + data2.predict + data3.predict + data4.predict + data5.predict + data6.predict + data7.predict + data8.predict + data9.predict + data10.predict).tolist()
print(len(pred))
print(pred)

best_pred = np.zeros((len(pred)))
best_score = 0
best_threshold = 0
for i in range(8):
    temp_pred = (np.array(pred) >= i).astype('int')
    print("verb threshold: {:d}".format(i))
    print("verb accuracy: {:.4f}".format(accuracy_score(data.label, temp_pred)))
    print("verb precision: {:.4f}".format(precision_score(data.label, temp_pred)))
    print("verb recall: {:.4f}".format(recall_score(data.label, temp_pred)))
    print("verb f1: {:.4f}".format(f1_score(data.label, temp_pred)))
    if f1_score(data.label, temp_pred) > best_score:
        best_score = f1_score(data.label, temp_pred)
        best_pred = temp_pred
        best_threshold = i

print("best verb threshold: {:d}".format(best_threshold))
print("best verb accuracy: {:.4f}".format(accuracy_score(data.label, best_pred)))
print("best verb precision: {:.4f}".format(precision_score(data.label, best_pred)))
print("best verb recall: {:.4f}".format(recall_score(data.label, best_pred)))
print("best verb f1: {:.4f}".format(f1_score(data.label, best_pred)))

data['predict'] = best_pred
print(data.predict.value_counts())
data[['id', 'predict']].to_csv('./submit/answer.txt', index=False, header=None)

data = pd.read_csv('./predict/TOEFI_verb_predict15.csv')
data2 = pd.read_csv('./predict/TOEFI_verb_predict14.csv')
data3 = pd.read_csv('./predict/TOEFI_verb_predict13.csv')
data4 = pd.read_csv('./predict/TOEFI_verb_predict12.csv')
data5 = pd.read_csv('./predict/TOEFI_verb_predict11.csv')
data6 = pd.read_csv('./predict/TOEFI_verb_predict10.csv')
data7 = pd.read_csv('./predict/TOEFI_verb_predict9.csv')
data8 = pd.read_csv('./predict/TOEFI_verb_predict8.csv')
data9 = pd.read_csv('./predict/TOEFI_verb_predict7.csv')
data10 = pd.read_csv('./predict/TOEFI_verb_predict6.csv')
data11 = pd.read_csv('./predict/TOEFI_verb_predict5.csv')
data12 = pd.read_csv('./predict/TOEFI_verb_predict4.csv')
data13 = pd.read_csv('./predict/TOEFI_verb_predict3.csv')
data14 = pd.read_csv('./predict/TOEFI_verb_predict2.csv')
data15 = pd.read_csv('./predict/TOEFI_verb_predict.csv')
data16 = pd.read_csv('./predict/TOEFI_verb_predict16.csv')
data17 = pd.read_csv('./predict/TOEFI_verb_predict17.csv')
data18 = pd.read_csv('./predict/TOEFI_verb_predict18.csv')
data19 = pd.read_csv('./predict/TOEFI_verb_predict19.csv')
data20 = pd.read_csv('./predict/TOEFI_verb_predict20.csv')
data21 = pd.read_csv('./predict/TOEFI_verb_predict21.csv')
data22 = pd.read_csv('./predict/TOEFI_verb_predict22.csv')
data23 = pd.read_csv('./predict/TOEFI_verb_predict23.csv')
data24 = pd.read_csv('./predict/TOEFI_verb_predict24.csv')
data25 = pd.read_csv('./predict/TOEFI_verb_predict25.csv')
data26 = pd.read_csv('./predict/TOEFI_verb_predict26.csv')
data27 = pd.read_csv('./predict/TOEFI_verb_predict27.csv')
data28 = pd.read_csv('./predict/TOEFI_verb_predict28.csv')
data29 = pd.read_csv('./predict/TOEFI_verb_predict29.csv')
data30 = pd.read_csv('./predict/TOEFI_verb_predict30.csv')
data31 = pd.read_csv('./predict/TOEFI_verb_predict31.csv')
data32 = pd.read_csv('./predict/TOEFI_verb_predict32.csv')
data33 = pd.read_csv('./predict/TOEFI_verb_predict33.csv')
data34 = pd.read_csv('./predict/TOEFI_verb_predict34.csv')

pred = (data11.predict + data12.predict + data13.predict
        + data14.predict + data15.predict + data17.predict + data18.predict
        + data21.predict + data22.predict
        + data26.predict + data29.predict + data30.predict2 + data31.predict3
        + data32.predict + data33.predict2
        ).tolist()

best_pred = (np.array(pred) >= 7).astype('int')
data['predict'] = best_pred
print(data.predict.value_counts())
data[['id', 'predict']].to_csv('./submit/answer.txt', index=False, header=None)

data = pd.read_csv('./predict/TOEFI_allpos_predict.csv')
data2 = pd.read_csv('./predict/TOEFI_allpos_predict2.csv')
data3 = pd.read_csv('./predict/TOEFI_allpos_predict3.csv')
data4 = pd.read_csv('./predict/TOEFI_allpos_predict4.csv')
data5 = pd.read_csv('./predict/TOEFI_allpos_predict5.csv')
data6 = pd.read_csv('./predict/TOEFI_allpos_predict6.csv')
data7 = pd.read_csv('./predict/TOEFI_allpos_predict7.csv')
data8 = pd.read_csv('./predict/TOEFI_allpos_predict8.csv')
data9 = pd.read_csv('./predict/TOEFI_allpos_predict9.csv')
data10 = pd.read_csv('./predict/TOEFI_allpos_predict10.csv')
data11 = pd.read_csv('./predict/TOEFI_allpos_predict11.csv')
data12 = pd.read_csv('./predict/TOEFI_allpos_predict12.csv')
data13 = pd.read_csv('./predict/TOEFI_allpos_predict13.csv')
data14 = pd.read_csv('./predict/TOEFI_allpos_predict14.csv')
data15 = pd.read_csv('./predict/TOEFI_allpos_predict15.csv')
data16 = pd.read_csv('./predict/TOEFI_allpos_predict16.csv')
data17 = pd.read_csv('./predict/TOEFI_allpos_predict17.csv')
data18 = pd.read_csv('./predict/TOEFI_allpos_predict18.csv')
data19 = pd.read_csv('./predict/TOEFI_allpos_predict19.csv')
data20 = pd.read_csv('./predict/TOEFI_allpos_predict20.csv')
data21 = pd.read_csv('./predict/TOEFI_allpos_predict21.csv')
data23 = pd.read_csv('./predict/TOEFI_allpos_predict22.csv')
data24 = pd.read_csv('./predict/TOEFI_allpos_predict23.csv')
data25 = pd.read_csv('./predict/TOEFI_allpos_predict24.csv')
data26 = pd.read_csv('./predict/TOEFI_allpos_predict25.csv')
data27 = pd.read_csv('./predict/TOEFI_allpos_predict26.csv')
data28 = pd.read_csv('./predict/TOEFI_allpos_predict27.csv')
data29 = pd.read_csv('./predict/TOEFI_allpos_predict28.csv')
data30 = pd.read_csv('./predict/TOEFI_allpos_predict29.csv')
data31 = pd.read_csv('./predict/TOEFI_allpos_predict30.csv')

pred = (data2.predict + data7.predict
        + data11.predict + data12.predict + data13.predict
        + data14.predict + data16.predict + data17.predict + data18.predict
        + data20.predict
        + data26.predict + data27.predict2 + data29.predict + data30.predict2 + data31.predict3
        ).tolist()

best_pred = (np.array(pred) >= 6).astype('int')
data['predict'] = best_pred
print(data.predict.value_counts())
data[['id', 'predict']].to_csv('./submit/answer.txt', index=False, header=None)
