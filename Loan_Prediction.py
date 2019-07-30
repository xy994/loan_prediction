#!/usr/bin/env python
# coding: utf-8

# # Homework 3 No Current
# #### Laura Chamberlain, Erik Chu, McCall James, Jui-Chuan Ma, Xinyang Yang

# In[1]:


get_ipython().run_line_magic('pylab', 'inline')
import os
from ipypublish import nb_setup
import pandas as pd
import numpy as np


# # Load and Clean Data

# In[2]:


df = pd.read_csv("loan.csv")


# In[3]:


df = df[df.loan_status!='Current']


# In[4]:


df.loan_status.value_counts()


# 0 - Charged Off, Default, Does not...:Charged Off, Late (31-120 days) <br>
# 1 - Fully Paid, Does not...:Fully Paid, Current

# In[5]:


df["Status"] = df.loan_status.apply(lambda x: 0 if ((x == "Charged Off" )|(x == "Default")|                                                    (x == "Does not meet the credit policy. Status:Charged Off")|                                                    (x == "Late (31-120 days)")) else 1)


# In[6]:


lst = ["Charged Off", "Current", "Default", "Does not meet the credit policy. Status:Charged Off",      "Does not meet the credit policy. Status:Fully Paid", "Fully Paid", "Late(31-120 days)"]

df = df[df.loan_status.isin(lst)]


# In[7]:


df.columns


# ### Drop Columns with over 50% NA

# In[8]:


df.mths_since_last_delinq.describe()


# In[9]:


df.isna().sum()[df.isna().sum() > len(df)/2]/len(df)


# In[10]:


df2 = df.drop(columns=(df.isna().sum()[df.isna().sum() > len(df)/2]/len(df)).index)


# In[11]:


df2.isna().sum()


# ### Select Related Numberic Columns

# In[12]:


df3 = df[['loan_amnt', 'funded_amnt', 'funded_amnt_inv','term', 'int_rate', 'installment', 'grade', 'sub_grade',       'home_ownership', 'annual_inc', 'verification_status',       'dti', 'delinq_2yrs','inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util',       'total_acc', 'initial_list_status', 'out_prncp', 'out_prncp_inv',       'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int',       'total_rec_late_fee', 'recoveries', 'collection_recovery_fee',       'last_pymnt_amnt', 'collections_12_mths_ex_med',  'application_type',       'acc_now_delinq', 'tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim',       'Status']].copy()


# In[13]:


df3['cr_history'] = (df.issue_d.astype(np.datetime64)-df.earliest_cr_line.astype(np.datetime64)).astype(int)                    /864000e+8


# In[14]:


df3.term = df3.term.map({' 36 months':36,' 60 months':60})


# In[15]:


df3.grade = df3.grade.map({'B':2, 'C':3, 'A':1, 'E':5, 'F':6, 'D':4, 'G':7})


# In[16]:


df3.initial_list_status = (df3.initial_list_status=='f')+0


# In[17]:


df2.application_type.value_counts()/len(df2)


# In[18]:


# recoveries have high corelated to loan_status
df3.drop(columns=['application_type', 'sub_grade','recoveries','collection_recovery_fee'],inplace=True)


# In[19]:


df3 = pd.get_dummies(df3, columns=['verification_status','home_ownership'],dtype=int)


# In[20]:


import seaborn as sns
cor = df3.corr()
#sns.heatmap(cor, xticklabels = cor.columns, yticklabels = cor.columns)
fig, ax = plt.subplots(figsize=(10,12))         # Sample figsize in inches
sns.heatmap(cor, linewidths=.1, ax=ax, cmap='plasma')


# ### Fill NAs

# In[21]:


df3.isna().sum()[df3.isna().sum()>0]


# In[22]:


sum(df3.isna().sum(axis=1)>0)


# In[23]:


df4 = df3.copy()


# In[24]:


df4 = df4[~df4.acc_now_delinq.isna()]


# In[25]:


df4.isna().sum()[df4.isna().sum()>0]


# In[26]:


df4["revol_util"] = df4["revol_util"].fillna(df4.annual_inc.mean())
df4["collections_12_mths_ex_med"] = df4["collections_12_mths_ex_med"].fillna(df4.collections_12_mths_ex_med.mean())
df4["tot_coll_amt"] = df4["tot_coll_amt"].fillna(df4.tot_coll_amt.mean())
df4["tot_cur_bal"] = df4["tot_cur_bal"].fillna(df4.tot_cur_bal.mean())
df4["total_rev_hi_lim"] = df4["total_rev_hi_lim"].fillna(df4.total_rev_hi_lim.mean())


# In[27]:


df4.isna().sum().sum()


# In[28]:


#df4.to_csv('df.csv')


# ---

# In[29]:


#df4 = pd.read_csv("df.csv")


# In[30]:


df4.shape


# In[31]:


X = df4._get_numeric_data().drop(columns = "Status")
Y = df4.Status
y = df4.Status


# In[32]:


import sklearn as sk #scikit learn
import sklearn.tree as tree
from IPython.display import Image  
import pydotplus
from sklearn.model_selection import train_test_split


# In[33]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30, random_state = 0)


# # Decision Tree

# In[33]:


Y.value_counts()/len(Y)


# In[34]:


dt = tree.DecisionTreeClassifier(max_depth = 3)


# In[35]:


dt.fit(X_train, y_train)


# In[36]:


dt_feature_names = list(X.columns)
dt_target_names = np.array(Y.unique(),dtype=np.str) 
tree.export_graphviz(dt, out_file='tree.dot', 
    feature_names=dt_feature_names, class_names=dt_target_names,
    filled=True)  
graph = pydotplus.graph_from_dot_file('tree.dot')
Image(graph.create_png())


# In[37]:


from sklearn.tree import DecisionTreeClassifier as CART
model = CART()
model.fit(X_train, y_train)
ypred = model.predict(X_test)


# In[38]:


probs = model.predict_proba(X_test)


# In[39]:


probs


# In[40]:


from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, auc


# In[41]:


#y_score = model.predict_proba(X)[:,1] 
y_score = model.predict_proba(X_test)[:,1] 
fpr, tpr, _ = roc_curve(y_test, y_score) 
title('ROC curve') 
xlabel('FPR (Precision)') 
ylabel('TPR (Recall)') 
plot(fpr,tpr) 
plot((0,1), ls='dashed',color='black') 
plt.show() 
print('Area under curve (AUC): ', auc(fpr,tpr))


# ---

# ### Logistic Regression

# In[39]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score

# instantiate a logistic regression model, and fit with X and y
model = LogisticRegression()
model = model.fit(X, y)

# check the accuracy on the training set
model.score(X, y)


# In[40]:


# Evaluate the model by splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
model2 = LogisticRegression()
model2.fit(X_train, y_train)


# In[41]:


# Predict class labels for the test set
predicted = model2.predict(X_test)
print(predicted)


# In[42]:


# Generate class probabilities
probs = model2.predict_proba(X_test)
print(probs)


# In[43]:


# Confusion Matrix
print(metrics.confusion_matrix(y_test, predicted))
print(metrics.classification_report(y_test, predicted))


# In[44]:


# generate evaluation metrics
print(metrics.accuracy_score(y_test, predicted))
print(metrics.roc_auc_score(y_test, probs[:, 1]))


# ### Discriminant Analysis

# In[45]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
model = LDA()
model.fit(X_train, y_train)


# In[46]:


#PREDICTION ON TEST DATA
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import confusion_matrix

y_hat = model.predict(X_test)


# In[47]:


#ACCURACY
#Out of sample
accuracy_score(y_test,y_hat)


# In[48]:


#CLASSIFICATION REPORT
print(classification_report(y_test, y_hat))


# In[49]:


#ROC, AUC
y_score = model.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_score)

title('ROC curve')
xlabel('FPR (Precision)')
ylabel('TPR (Recall)')

plot(fpr,tpr)
plot((0,1), ls='dashed',color='black')
plt.show()
print('Area under curve (AUC): ', auc(fpr,tpr))


# In[50]:


#CONFUSION MATRIX
cm = confusion_matrix(y_test, y_hat)
cm


# ### Naive_Bayes

# In[51]:


#FIT MODEL
from sklearn.naive_bayes import GaussianNB as NB
model = NB()
model.fit(X_train,y_train)


# In[52]:


#CONFUSION MATRIX
ypred = model.predict(X_test)
cm = confusion_matrix(y_test, ypred)
cm


# In[53]:


#ACCURACY
accuracy_score(y_test,ypred)


# In[54]:


#CLASSIFICATION REPORT
print(classification_report(y_test, ypred))


# In[55]:


#ROC, AUC
y_score = model.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_score)

title('ROC curve')
xlabel('FPR (Precision)')
ylabel('TPR (Recall)')

plot(fpr,tpr)
plot((0,1), ls='dashed',color='black')
plt.show()
print('Area under curve (AUC): ', auc(fpr,tpr))


# In[56]:


mean(Y)


# ### SMOTE

# In[47]:


from imblearn.over_sampling import SMOTE


# In[48]:


sm = SMOTE(random_state=2)
X_res, y_res = sm.fit_sample(X, y.ravel())


# In[49]:


mean(y_res)


# In[50]:


sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())
X_test_res, y_test_res = sm.fit_sample(X_test, y_test.ravel())


# In[51]:


import sklearn as sk #scikit learn
import sklearn.tree as tree
from IPython.display import Image  
import pydotplus
from sklearn.model_selection import train_test_split


# In[52]:


dt = tree.DecisionTreeClassifier(max_depth = 3)


# In[53]:


dt.fit(X_train_res, y_train_res)


# In[54]:


dt_feature_names = list(X.columns)
dt_target_names = np.array(Y.unique(),dtype=np.str) 
tree.export_graphviz(dt, out_file='tree.dot', 
    feature_names=dt_feature_names, class_names=dt_target_names,
    filled=True)  
graph = pydotplus.graph_from_dot_file('tree.dot')
Image(graph.create_png())


# In[55]:


from sklearn.tree import DecisionTreeClassifier as CART
model = CART()
model.fit(X_train_res, y_train_res)
ypred = model.predict(X_test_res)


# In[56]:


probs = model.predict_proba(X_test)


# In[57]:


from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, auc


# In[58]:


#y_score = model.predict_proba(X)[:,1] 
y_score = model.predict_proba(X)[:,1] 
fpr, tpr, _ = roc_curve(y, y_score) 
title('ROC curve') 
xlabel('FPR (Precision)') 
ylabel('TPR (Recall)') 
plot(fpr,tpr) 
plot((0,1), ls='dashed',color='black') 
plt.show() 
print('Area under curve (AUC): ', auc(fpr,tpr))


# ### KNN

# In[59]:


len(df4.columns)


# In[60]:


#FIT MODEL
from sklearn.neighbors import KNeighborsClassifier as KNNC
model = KNNC(n_neighbors=3, algorithm='ball_tree')
model.fit(X_train, y_train)


# In[61]:


#CONFUSION MATRIX
ypred = model.predict(X_test)
cm = confusion_matrix(y_test, ypred)
cm


# In[62]:


#ACCURACY
accuracy_score(y_test,ypred)


# In[63]:


#CLASSIFICATION REPORT
print(classification_report(y_test, ypred))


# In[64]:


#ROC, AUC
y_score = model.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_score)

title('ROC curve')
xlabel('FPR (Precision)')
ylabel('TPR (Recall)')

plot(fpr,tpr)
plot((0,1), ls='dashed',color='black')
plt.show()
print('Area under curve (AUC): ', auc(fpr,tpr))


# ### Support Vector Machines

# In[ ]:


#FIT MODEL
from sklearn import svm
model = svm.SVC()
model.fit(X_train,y_train)
ypred = model.predict(X_test)


# In[ ]:


#CONFUSION MATRIX
cm = confusion_matrix(y, ypred)
cm


# In[ ]:


#ACCURACY
accuracy_score(y,ypred)


# In[ ]:


# Predict class labels for the test set
predicted = model.predict(X_test)


# In[ ]:


# Generate class probabilities
probs = model.predict_proba(X_test)


# In[ ]:


# Confusion Matrix
print(metrics.confusion_matrix(y_test, predicted))
print(metrics.classification_report(y_test, predicted))


# In[ ]:


# generate evaluation metrics
print(metrics.accuracy_score(y_test, predicted))
print(metrics.roc_auc_score(y_test, probs[:, 1]))


# #ROC, AUC
# y_score = model.predict_proba(X_test)[:,1]
# fpr, tpr, _ = roc_curve(y_test, y_score)
# 
# title('ROC curve')
# xlabel('FPR (Precision)')
# ylabel('TPR (Recall)')
# 
# plot(fpr,tpr)
# plot((0,1), ls='dashed',color='black')
# plt.show()
# print('Area under curve (AUC): ', auc(fpr,tpr))

# ### SMOTE Random Forest

# In[34]:


from imblearn.over_sampling import SMOTE


# In[35]:


sm = SMOTE(random_state=2)
X_res, y_res = sm.fit_sample(X, y.ravel())


# In[36]:


mean(y_res)


# In[37]:


sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())
X_test_res, y_test_res = sm.fit_sample(X_test, y_test.ravel())


# In[38]:


from sklearn.ensemble import RandomForestClassifier


# In[39]:


dt = RandomForestClassifier(random_state = 0)


# In[40]:


dt.fit(X_train_res, y_train_res)


# In[42]:


ypred = dt.predict(X_test_res)


# In[43]:


probs = dt.predict_proba(X_test)


# In[44]:


from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, auc


# In[46]:


#y_score = model.predict_proba(X)[:,1] 
y_score = dt.predict_proba(X)[:,1] 
fpr, tpr, _ = roc_curve(y, y_score) 
title('ROC curve') 
xlabel('FPR (Precision)') 
ylabel('TPR (Recall)') 
plot(fpr,tpr) 
plot((0,1), ls='dashed',color='black') 
plt.show() 
print('Area under curve (AUC): ', auc(fpr,tpr))


# In[ ]:




