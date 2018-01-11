import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

clsr_names=["Nearest Neighbors", "Linear SVM", "RBF SVM",
            "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
            "Naive Bayes"]

classifiers = [KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB()]

df01=pd.read_csv('bank-additional/bank-additional-full.csv', sep=';',header=0)
df02=df01.dropna(axis=1, how='all')
df=df02.dropna(axis=0, how='any')
cols=df.dtypes
colnms=df.columns
i=0
cat_cols=[]
for eachcol in cols:
    if eachcol.name=="object":
        cat_cols.append(colnms[i])
    i+=1

df1=pd.get_dummies(df,columns=cat_cols)
n=len(df1.index)
m=len(df1.columns)
x_all=df1.iloc[:,0:(m-2)]
y_all=df1['y_yes']

x_trn, x_tst, y_trn, y_tst = train_test_split(x_all, y_all, test_size=0.8, random_state=42)
scaler = MinMaxScaler()
scaler.fit(x_trn)
x_trn_n=scaler.transform(x_trn)
x_tst_n=scaler.transform(x_tst)
clf = classifiers[1]
model=clf.fit(x_trn_n,y_trn)
y_pred=model.predict(x_tst_n)
acc1=float((y_pred==y_tst).sum())/float(len(y_tst))
print("Linear SVM accuracy: {0:.3f}%".format(acc1))
weight=model.coef_[0]
var2wgt=pd.DataFrame(list(zip(list(df1),weight)),columns=['variable','weight'])
var2wgt_sorted=var2wgt.reindex(var2wgt.weight.abs().sort_values(ascending=False).index)
print("Top 10 weighted variables:")
print(var2wgt_sorted[0:10])

var_names=list(var2wgt_sorted['variable'][0:10])
var_imp=list(var2wgt_sorted['weight'][0:10].abs())
y_pos = np.arange(len(var_names),0,-1)
fig = plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.barh(y_pos, var_imp, align='center', alpha=0.5)
plt.yticks(y_pos, var_names)
plt.xlabel('Weight')
plt.title('Linear SVM')
plt.ylim(0,11)

clf=classifiers[4]
model=clf.fit(x_trn_n,y_trn)
y_pred=model.predict(x_tst_n)
acc2=float((y_pred==y_tst).sum())/float(len(y_tst))
print("Random forest accuracy: {0:.3f}%".format(acc2))
imp=model.feature_importances_
var2imp=dict(zip(list(df1),imp))
var2imp_sorted=pd.DataFrame(columns=['variable','weight'])
for key in sorted(var2imp, key=lambda k:abs(var2imp[k]),reverse=True):
    temp=pd.DataFrame([[key,var2imp[key]]],columns=['variable','weight'])
    var2imp_sorted=var2imp_sorted.append(temp)
print("Top 10 important variables:")
print(var2imp_sorted[0:10])

var_names=list(var2imp_sorted['variable'][0:10])
var_imp=list(var2imp_sorted['weight'][0:10])
y_pos = np.arange(len(var_names),0,-1)
plt.subplot(1, 2, 2)
plt.barh(y_pos, var_imp, align='center', alpha=0.5)
plt.yticks(y_pos, var_names)
plt.xlabel('Weight')
plt.title('Random Forest')
plt.ylim(0,11)
plt.tight_layout()
fig.savefig('plot.png',dpi=400)
'''
print("Comparing different models:")
for name, clf in zip(clsr_names, classifiers):
    model=clf.fit(x_trn_n,y_trn)
    y_pred=model.predict(x_tst_n)
    print(name+" Accuracy: {0:.3f}%".format(float((y_pred==y_tst).sum())/float(len(y_tst))))
'''
