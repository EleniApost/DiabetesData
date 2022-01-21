#!/usr/bin/env python
# coding: utf-8

# In[157]:


# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# Common imports
import numpy as np
import os
import pandas as pd

# to make this notebook's output stable across runs
np.random.seed(1)

# To plot pretty figures
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
import seaborn as sns

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "classification"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    
    
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import  OrdinalEncoder #We use the ordinal encoder for the labels as there are only two values and it will work for what we need. 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score,auc
import statistics as stats
import warnings
warnings.filterwarnings("ignore")
import tensorflow.keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, History, LearningRateScheduler
from tensorflow.keras import regularizers
# from tensorflow.keras.layers import Recurrent_Dropout
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers 
import time
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


# ## Importing the data

# In[3]:


df=pd.read_csv('diabetes_data.csv')


# ## Data exploration

# Exploring our data and checking for missing values and outliers

# In[6]:


df.head()


# ### Checking for missing values

# In[5]:


df.info() 


# We can see that there are no null values. 

# In[7]:


# Looking at the age distribution
df.describe()


# ### Checking for outliers

# In[8]:


df.Age.value_counts()


# In[9]:


attr_categorical = ["Gender","Polyuria","Polydipsia", "sudden weight loss","weakness","Polyphagia","Genital thrush",
                    "visual blurring","Itching","Irritability","delayed healing","partial paresis","muscle stiffness","Alopecia","Obesity","class"]
attr_num  = ["Age"]


# In[10]:


### Data distribution


# In[11]:


for label in attr_categorical:
    
    print(df[label].value_counts(normalize=True))


# In[12]:



import matplotlib.pyplot as plt
df.hist(bins=50, figsize=(20,15))
plt.show()


# In[13]:




fig, ax =plt.subplots(5,4, figsize=(24
                                    , 26))
   
sns.set( {'axes.labelsize' : 16 })
sns.countplot(ax = ax[0,0], x = df['Gender'])

sns.countplot(ax = ax[0,1], x = df['Polyuria'])

sns.countplot(ax = ax[0,2], x = df['Polydipsia'])

sns.countplot(ax = ax[0,3], x = df['sudden weight loss'])

sns.countplot(ax = ax[1,0], x = df['weakness'])

sns.countplot(ax = ax[1,1], x = df['Polyphagia'])

sns.countplot(ax = ax[1,2], x = df['Genital thrush'])

sns.countplot(ax = ax[1,3], x = df['visual blurring'])

sns.countplot(ax = ax[2,0], x = df['Itching'])

sns.countplot(ax = ax[2,1], x = df['Irritability'])

sns.countplot(ax = ax[2,2], x = df['delayed healing'])

sns.countplot(ax = ax[2,3], x = df['partial paresis'])

sns.countplot(ax = ax[3,0], x = df['muscle stiffness'])

sns.countplot(ax = ax[3,1], x = df['Alopecia'])

sns.countplot(ax = ax[3,2], x = df['Obesity'])
sns.histplot(ax=ax[3,3], x=df['Age'])
sns.countplot(ax = ax[4,0], x = df['class'])

fig.suptitle('Data distribution', y =0.9, size = 16)  

plt.savefig('data_vis.png')


# ### Checking for imbalanced data

# In[14]:


ax = sns.countplot(x = df["class"])  
plt.show()


# We can see that the minority class os approximately 38% of the data, so we have a mildly imbalanced dataset. 
# https://developers.google.com/machine-learning/data-prep/construct/sampling-splitting/imbalanced-data

# ## Data preparation

# ### Handling Categorical data

# In[15]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import  OrdinalEncoder #We use the ordinal encoder for the labels as there are only two values and it will work for what we need. 
ordinal_encoder = OrdinalEncoder()
num_pipeline = Pipeline([('std_scaler', StandardScaler()) # Scaling numerical values
 ])




full_pipeline = ColumnTransformer([
        ("num", num_pipeline, attr_num), #Scaling the data with Standard scalar
        ("cat", ordinal_encoder, attr_categorical), # Transforming categorical data  
    ])

data_prepared = full_pipeline.fit_transform(df)

data = pd.DataFrame(data_prepared, columns = df.columns)


# ## Looking for colinearity and correlations 

# In[16]:


corr_matrix = data.corr(method = "pearson")


# In[17]:


plt.figure(figsize=(15,15))
sns.heatmap(corr_matrix,square=True,annot=True,cmap= 'twilight_shifted')

plt.title('Pearson Correlation Matrix', y =1.01, size = 16)  

plt.savefig('Pearson correlation.png')


# In[18]:


corr_matrix = data.corr(method = "spearman")


# In[19]:


plt.figure(figsize=(15,15))
plt.title('Spearman Correlation Matrix', y =1.01, size = 16) 
sns.heatmap(corr_matrix,square=True,annot=True,cmap= 'twilight_shifted')


# There is a moderetely increased correlation between polyuria and polydipsia which makes sense as increased thirst would lead to increased urination. Overall there are no stong correlations in our dataset. 

# ## Preparing the data for Machine learning algorithms
# ### Splitting the data

# In[20]:


from sklearn.model_selection import train_test_split


# In[21]:


data_labels = data['class']
data_p = data.drop('class', axis = 1)


# In[22]:


df_train, df_test,train_labels,test_labels = train_test_split(data_p, data_labels, train_size = 0.8)


# In[23]:


df_train.shape, train_labels.shape, df_test.shape, test_labels.shape


# In[24]:


##### Loading the Machne learning models


# In[25]:


forest = RandomForestClassifier()

knn = KNeighborsClassifier()
XGB = XGBClassifier()
voting_clf = VotingClassifier(
 estimators=[('forest', forest), ('knn', knn), ('xgb', XGB)],
 voting='soft'
 )


# In[27]:


smote= SMOTE()
df_train,train_labels = smote.fit_resample(df_train,train_labels)
df_train.shape, train_labels.shape, df_test.shape, test_labels.shape


# In[28]:


classifiers = [forest,knn,XGB,voting_clf]
from sklearn.metrics import plot_confusion_matrix


# ### Classification report and confusion matrices

# We run all classifiers once to estimate their performance

# In[34]:


for clf in classifiers:
    clf.fit(df_train,train_labels)
    y_pred = clf.predict(df_test)
    f_probs = clf.predict_proba(df_test)[:, 1]
    clf_auc = roc_auc_score(test_labels,f_probs)
    print('Train accuracy: {}'.format(clf.score(df_train, train_labels)))
    print('Test accuracy: {}'.format(clf.score(df_test, test_labels)))
    print('AUC score: ',(clf_auc))
    print('For Classifier',clf,classification_report(test_labels, y_pred,digits=3))


# In[35]:


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,10))

for cls, ax in zip(classifiers, axes.flatten()):
    plot_confusion_matrix(cls, 
                          df_test, 
                          test_labels, 
                          ax=ax, 
                          cmap='Blues',
                         display_labels=['Negative','Positive'])
    ax.title.set_text(type(cls).__name__)
    ax.grid(False)
    plt.grid(False)

plt.tight_layout()

#plt.show()

plt.savefig('conf_matrix.png')


# In[37]:


#MLNN first run
np.random.seed(1)
smote= SMOTE()
X_train, X_test, y_train, y_test = train_test_split(data_p, data_labels, train_size = 0.8, shuffle=True)
X_train,y_train = smote.fit_resample(X_train,y_train)
input_dim = 16
num_classes = 1
num_epochs = 1500
layer_dim = 1
learning_rate = 0.001
batch_size =32#int(opt)32
dropout=0.05
hidden_dim=512
output_activation = 'sigmoid'
initial_activation = 'relu'
kernel_initializer="glorot_uniform"
bias_initializer="glorot_uniform"
loss_function=tf.keras.losses.BinaryCrossentropy()
optimiser=optimizers.Adamax(lr=learning_rate)
metric=tf.keras.metrics.AUC()#,tf.keras.metrics.Accuracy()
kernel_regularizer='l2' # ,'l1'

##Building the MLP
        
history = History()
model = Sequential()

#Add input layer
model.add(Dense(hidden_dim, input_dim=input_dim, activation=initial_activation, use_bias=True, bias_initializer=bias_initializer, kernel_initializer=kernel_initializer,kernel_regularizer=kernel_regularizer))
model.add(Dropout(dropout))
#Add hidden layer
model.add(Dense(hidden_dim, activation=initial_activation, use_bias=False, bias_initializer=bias_initializer, kernel_initializer=kernel_initializer,kernel_regularizer=kernel_regularizer))
#Add output layer
model.add(Dense(units=num_classes, activation=output_activation))
   
sgd=optimizers.Adamax(lr=learning_rate)     
model.compile(loss=loss_function,
              optimizer=optimiser,
              metrics=[metric,'accuracy']) 
print(model.summary())
model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs, verbose=1, callbacks=[history])
plt.plot(history.history['loss'], label='train')
plt.show()

#Model Evaluation
score =(loss, AUC,accuracy) = model.evaluate(X_test, y_test, verbose=1)
#y_pred=model.predict(X_test)
#print('Test AUC (from sklearn):',custom_auc(y_test, y_pred))
print('Test loss:',score[0])
print('Test AUC (from tf):',score[1])
print('Test accuracy:',score[2])
print(model.evaluate(X_test, y_test, verbose=1))
#print('Test f1-score:',score[3])
#print('Test precision:',score[4])
#print('Test recall:',score[5])


# In[46]:


mnn_pred = model.predict_classes(X_test)


# In[47]:


# precision tp / (tp + fp)
precision = precision_score(y_test, mnn_pred)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_test, mnn_pred)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, mnn_pred)
print('F1 score: %f' % f1)


# In[48]:


cm = confusion_matrix(y_true=y_test, y_pred=mnn_pred.round())


# In[49]:


#Confusion map for MLNN
import itertools
from matplotlib.pyplot import figure
def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize = (8,6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    
    
   
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")
    ax.grid(False)
    plt.grid(False)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.gcf().subplots_adjust(bottom=0.15)

    plt.savefig('mnn_conf_matrix.png')
    
plot_confusion_matrix(cm=cm, classes=['Negative','Positive'], title='MNN')


# ## Feature importance and correlation

# In[50]:


importances = forest.feature_importances_


# In[51]:


cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.feature_names_in_)
attributes = attr_num + cat_one_hot_attribs
sorted(zip(importances, attributes), reverse=True) # Displays importance scores next to attribute names. 


# In[55]:


attributes


# In[152]:


features = pd.DataFrame({"Importances":importances,
                  "Feature":attributes})
features_sorted_desc= features.sort_values('Importances',ascending=False)

fig = plt.figure()
plt.figure(figsize = (12,6))
plt.subplots_adjust(left=0.3)
plt.barh('Feature','Importances',data = features_sorted_desc)
plt.title('')
plt.ylabel('')
plt.xlabel('Feature Importance')
plt.title('Feature Importance', y =1.01, size = 16)

plt.yticks(fontsize=14)
fig.suptitle('', fontsize=14, fontweight='bold')
ax = plt.axes()

ax.set_facecolor('white')
ax.set_ylabel('Features', fontsize=16)

plt.savefig('Feature Importance .png')


# In[57]:


xg_importances = XGB.feature_importances_


# In[153]:


features = pd.DataFrame({"Importances":xg_importances,
                  "Feature":attributes})
features_sorted_desc= features.sort_values('Importances',ascending=False)

fig = plt.figure()
plt.figure(figsize = (12,6))
plt.subplots_adjust(left=0.3)
plt.barh('Feature','Importances',data = features_sorted_desc)
plt.title('')
plt.ylabel('')
plt.xlabel('Feature Importance')
plt.title('Feature Importance', y =1.01, size = 16)

plt.yticks(fontsize=14)
fig.suptitle('', fontsize=14, fontweight='bold')
ax = plt.axes()

ax.set_facecolor('white')
ax.set_ylabel('Features', fontsize=16)

plt.savefig('Feature Importance_xg .png')


# In[70]:


corr  = []
for feature in data_p:
    corr.append(np.corrcoef(data_p[feature],data_labels)[0][1])
    print(feature, np.corrcoef(data_p[feature],data_labels)[0][1])


# In[146]:


features = pd.DataFrame({"Pearson Corr":corr,
                  "Feature":attributes})
features_sorted_desc= features.sort_values('Pearson Corr',ascending=False)
fig = plt.figure()
plt.figure(figsize = (12,6))
plt.subplots_adjust(left=0.3)
plt.barh('Feature','Pearson Corr',data = features_sorted_desc)
plt.title('')
plt.ylabel('')
plt.xlabel('Pearson Corr')
plt.title('Pearson correlation ', y =1.01, size = 16)
plt.yticks(fontsize=14)
fig.suptitle('', fontsize=14, fontweight='bold')
ax = plt.axes()
ax.set_facecolor('white')
ax.set_ylabel('Features', fontsize=16)
plt.savefig('Pearson Correlation .png')


# # Estimating the best models

# In[154]:


stat_forest = []
stat_knn = []
stat_XGB = []
AUC_forest = []
AUC_knn = []
AUC_XGB = []


# In[158]:


mf = []

forest = RandomForestClassifier(random_state=1)
for times in range(5):
    df_train, df_test,train_labels,test_labels = train_test_split(data_p, data_labels, train_size = 0.8, shuffle=True)
    df_train,train_labels = smote.fit_resample(df_train,train_labels)
    param_grid = [
    # try 12 (3×4) combinations of hyperparameters
     # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [ 50,100], 'max_features': [2,3, 4], 'criterion': ['gini','entropy'],'max_depth' : [10,20]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [50,100], 'max_features': [2, 3, 4],'criterion': ['gini','entropy'],'max_depth' : [10,20]},
  ]
    grid = GridSearchCV(forest,param_grid, cv=10,return_train_score=True) 
    
    
    

    
    grid.fit(df_train,train_labels)
    forest = grid.best_estimator_
    f_probs = forest.predict_proba(df_test)[:, 1]
    forest_auc = roc_auc_score(test_labels,f_probs)
    stat_forest.append(forest.score(df_test, test_labels))
    
    AUC_forest.append(forest_auc)
    mf.append(forest)
print("  Mean AUC ", round(stats.mean(AUC_forest),3),"  Mean ACC ", round(stats.mean(stat_forest),3))    


# In[159]:


t = zip(mf,AUC_forest, stat_forest)
print(list(t))


# In[161]:


kn = []
knn = KNeighborsClassifier()
for times in range(5):
    df_train, df_test,train_labels,test_labels = train_test_split(data_p, data_labels, train_size = 0.8, shuffle=True)
    df_train,train_labels = smote.fit_resample(df_train,train_labels)    
    param_grid = [
        # try 12 (3×4) combinations of hyperparameters
        {'n_neighbors': [ 3,5,11,19], 'weights': ['uniforn','distance'], 'metric': ['minkowski','manhattan','euclidean']},
        # then try 6 (2×3) combinations with bootstrap set as False
        #{'bootstrap': [False], 'n_estimators': [50,100,200], 'max_features': [2, 3, 4],'criterion': ['gini','entropy']},
     ]


    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
    grid = GridSearchCV(knn, param_grid, cv=20, return_train_score=True)                       

    grid.fit(df_train,train_labels)

    knn2 = grid.best_estimator_
    g_probs = knn2.predict_proba(df_test)[:, 1]
    knn_auc = roc_auc_score(test_labels,g_probs)   
    stat_knn.append(knn2.score(df_test, test_labels))
    
    AUC_knn.append(knn_auc)
    kn.append(knn2)

k = zip(kn,AUC_knn, stat_knn)
print(list(k))


# In[162]:


xg = []
XGB = XGBClassifier(use_label_encoder=False, eval_metric = 'auc')
for times in range(10):
    df_train, df_test,train_labels,test_labels = train_test_split(data_p, data_labels, train_size = 0.8, shuffle=True)
    df_train,train_labels = smote.fit_resample(df_train,train_labels)    
    param_grid = [
         
    { 'gamma': [0,1,5],
              'learning_rate': [0.01, 0.1],
              'max_depth': [10,20],
              'n_estimators': [50,100],
              }
 ]


    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
    XGBCL = GridSearchCV(XGB, param_grid, cv=10, return_train_score=True)                       

    XGBCL.fit(df_train,train_labels)

    XGB = XGBCL.best_estimator_
    g_probs = XGB.predict_proba(df_test)[:, 1]
    XGB_auc = roc_auc_score(test_labels,g_probs)   
    stat_XGB.append(XGB.score(df_test, test_labels))
    
    AUC_XGB.append(XGB_auc)
    xg.append(XGB)

x = zip(xg,AUC_XGB, stat_XGB)
print(list(x))


# In[164]:


from prettytable                     import PrettyTable
from astropy.table                   import Table, Column


# # Running the models 100-200 times to evaluate overall performance

# #### Machine learning models

# In[165]:




forest = RandomForestClassifier(random_state = 1,bootstrap = False, criterion= 'gini',max_depth = 10, max_features= 2, n_estimators= 50)

knn = KNeighborsClassifier(metric = 'manhattan', n_neighbors= 3, weights = 'distance')

XGB = XGBClassifier(use_label_encoder=False, eval_metric = 'auc',gamma= 0, learning_rate= 0.1, max_depth= 10, n_estimators= 100)
#Voting classifier on all the best models
voting_clf = VotingClassifier(
 estimators=[('forest', forest), ('knn', knn), ('xgb', XGB)],
 voting='soft'
 )
stat_forest = []
stat_knn = []
stat_XGB = []
stat_vc = []
AUC_forest = []
AUC_knn = []
AUC_XGB = []
AUC_vc = []

for times in range(300):
    df_train, df_test,train_labels,test_labels = train_test_split(data_p, data_labels, train_size = 0.8, shuffle=True)
    df_train,train_labels = smote.fit_resample(df_train,train_labels)
    forest.fit(df_train,train_labels)
    f_probs = forest.predict_proba(df_test)[:, 1]
    forest_auc = roc_auc_score(test_labels,f_probs) 
    
    #print('Forest:')
    #print('Train accuracy: {}'.format(forest.score(df_train, train_labels)))
    #print('Test accuracy: {}'.format(forest.score(df_test, test_labels)))
     
    knn.fit(df_train,train_labels)
    k_probs = knn.predict_proba(df_test)[:, 1]
    knn_auc = roc_auc_score(test_labels,k_probs)
    #print('KNN:')
    #print('Train accuracy: {}'.format(knn.score(df_train, train_labels)))
    #print('Test accuracy: {}'.format(knn.score(df_test, test_labels)))
      
    XGB.fit(df_train,train_labels)
    XGB_pred = XGB.predict(df_test)
    X_probs = knn.predict_proba(df_test)[:, 1]
    XGB_auc = roc_auc_score(test_labels,X_probs)
    #accuracy = accuracy_score(test_labels, y_pred)
    #print('XGBoost:')
    #print('Train accuracy: {}'.format(XGB.score(df_train, train_labels)))
    #print('Test accuracy: {}'.format(XGB.score(df_test, test_labels)))
    
    voting_clf.fit(df_train,train_labels)
    vc_pred = voting_clf.predict(df_test)
    vc_probs = voting_clf.predict_proba(df_test)[:, 1]
    vc_auc = roc_auc_score(test_labels,vc_probs)
    
    
    stat_forest.append(forest.score(df_test, test_labels))
    stat_knn.append(knn.score(df_test, test_labels))
    stat_XGB.append(accuracy_score(test_labels, XGB_pred))
    stat_vc.append(voting_clf.score(df_test, test_labels))
    
    AUC_forest.append(forest_auc)
    AUC_knn.append(knn_auc)
    AUC_XGB.append(XGB_auc)
    AUC_vc.append(vc_auc)
    
        


# In[166]:


from prettytable                     import PrettyTable
from astropy.table                   import Table, Column
Model_Table = PrettyTable()
Model_Table.field_names = [" ", "   Random forest Classififier  ", "     K-Nearest Neighbours       ", "     XGBoost       ", "Voting classifier"]
Model_Table.add_row(["  Max  ", round(max(stat_forest),3), round(max(stat_knn),3), round(max(stat_XGB),3),round(max(stat_vc),3)])
Model_Table.add_row(["  Min  ", round(min(stat_forest),3),round(min(stat_knn),3),round(min(stat_XGB),3),round(min(stat_vc),3)])
Model_Table.add_row(["  Mean  ", round(stats.mean(stat_forest),3),round(stats.mean(stat_knn),3),round(stats.mean(stat_XGB),3),round(stats.mean(stat_vc),3)])
Model_Table.add_row(["  StDev  ", round(stats.stdev(stat_forest),3),round(stats.stdev(stat_knn),3),round(stats.stdev(stat_XGB),3),round(stats.stdev(stat_vc),3)])
print("Detailed accuracy performance of all models:")
print(Model_Table)#


# In[167]:


Model_Table2 = PrettyTable()
Model_Table2.field_names = [" ", "   Random forest Classififier  ", "     K-Nearest Neighbours       ", "     XGBoost       ", "Voting Classifier"]
Model_Table2.add_row(["  Max  ", round(max(AUC_forest),3), round(max(AUC_knn),3), round(max(AUC_XGB),3),round(max(AUC_vc),3)])
Model_Table2.add_row(["  Min  ", round(min(AUC_forest),3),round(min(AUC_knn),3),round(min(AUC_XGB),3),round(min(AUC_vc),3)])
Model_Table2.add_row(["  Mean  ", round(stats.mean(AUC_forest),3),round(stats.mean(AUC_knn),3),round(stats.mean(AUC_XGB),3),round(stats.mean(AUC_vc),3)])
Model_Table2.add_row(["  StDev  ", round(stats.stdev(AUC_forest),3),round(stats.stdev(AUC_knn),3),round(stats.stdev(AUC_XGB),3),round(stats.stdev(AUC_vc),3)])

print("Detailed AUC performance of all models:")
print(Model_Table2)#


# ### Multilayer Neural Network

# In[ ]:



#Neural Network Hyperparameters

np.random.seed(1)
ml_loss = []
ml_auc = []
ml_accuracy = []
for times in range(25):
    X_train, X_test, y_train, y_test = train_test_split(data_p, data_labels, train_size = 0.8, shuffle=True)
    X_train,y_train = smote.fit_resample(X_train,y_train)
    input_dim = 16
    num_classes = 1
    num_epochs = 1000
    layer_dim = 1
    learning_rate = 0.001
    batch_size =32#int(opt)32
    dropout=0.05
    hidden_dim=512
    output_activation = 'sigmoid'
    initial_activation = 'relu'
    kernel_initializer="glorot_uniform"
    bias_initializer="glorot_uniform"
    loss_function=tf.keras.losses.BinaryCrossentropy()
    optimiser=optimizers.Adamax(lr=learning_rate)
    metric=tf.keras.metrics.AUC()#,tf.keras.metrics.Accuracy()
    kernel_regularizer='l2'#,'l1_l2'
    
    ##Building the MLP
            
    history = History()
    model = Sequential()
    
    #Add input layer
    model.add(Dense(hidden_dim, input_dim=input_dim, activation=initial_activation, use_bias=True, bias_initializer=bias_initializer, kernel_initializer=kernel_initializer,kernel_regularizer=kernel_regularizer))
    model.add(Dropout(dropout))
    #Add hidden layer
    model.add(Dense(hidden_dim, activation=initial_activation, use_bias=True, bias_initializer=bias_initializer, kernel_initializer=kernel_initializer,kernel_regularizer=kernel_regularizer))
    model.add(Dropout(dropout))
    #Add hidden layer
    model.add(Dense(hidden_dim, activation=initial_activation, use_bias=False, bias_initializer=bias_initializer, kernel_initializer=kernel_initializer,kernel_regularizer=kernel_regularizer))
    
    #Add output layer
    model.add(Dense(units=num_classes, activation=output_activation))
       
    sgd=optimizers.Adamax(lr=learning_rate)     
    model.compile(loss=loss_function,
                  optimizer=optimiser,
                  metrics=[metric,'accuracy']) 
    print(model.summary())
    callback = tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience = 300)
    model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs, verbose=0, callbacks=[callback, history])
    plt.plot(history.history['loss'], label='train')
    plt.show()
    
    #Model Evaluation
    score =(loss, AUC,accuracy) = model.evaluate(X_test, y_test, verbose=0)
    #y_pred=model.predict(X_test)
    #print('Test AUC (from sklearn):',custom_auc(y_test, y_pred))
    ml_loss.append(score[0])
    ml_auc.append(score[1])
    ml_accuracy.append(score[2])
    #print('Test loss:',score[0])
    #print('Test AUC (from tf):',score[1])
    #print('Test accuracy:',score[2])
# calculate testing accuracy
#print(ml_loss,ml_auc,ml_accuracy)

Model_Table = PrettyTable()
Model_Table.field_names = [" ", "   Loss function  ", "     AUC     ", "    Accuracy      "]
Model_Table.add_row(["  Max  ", round(max(ml_loss),3), round(max(ml_auc),3), round(max(ml_accuracy),3)])
Model_Table.add_row(["  Min  ", round(min(ml_loss),3),round(min(ml_auc),3),round(min(ml_accuracy),3)])
Model_Table.add_row(["  Mean  ", round(stats.mean(ml_loss),3),round(stats.mean(ml_auc),3),round(stats.mean(ml_accuracy),3)])
Model_Table.add_row(["  StDev  ", round(stats.stdev(ml_loss),3),round(stats.stdev(ml_auc),3),round(stats.stdev(ml_accuracy),3)])
print("Detailed accuracy performance of MLNN:")
print(Model_Table)#


# # Feature Selection

# In[90]:


#Forest parameter 
from sklearn.feature_selection import SelectFromModel


# From pearson correlation coefficient we had that itching is the least correlated with our outcome and from feature selection we had that weakness and obesity are least important. We will try and remove each of those and see how the Randon Forest classifier will perform, as it is the algorythm with the best performance.

# In[58]:


forest = RandomForestClassifier(random_state=1)
# Drop itching
data_sel = data_p.drop('Itching', axis = 1)
#drop weakness
data_sel1 = data_p.drop('weakness', axis = 1)
#drop itching and delayed healing
data_sel2 = data_p.drop(['weakness','Obesity'], axis = 1)
#drop itching, delayed healing and obesity 
data_sel3 = data_p.drop(['Itching','delayed healing', 'Obesity'], axis = 1)


# In[67]:


stat_forest = []

AUC_forest = []
forest = RandomForestClassifier(random_state=1, criterion= 'gini', max_features= 2, n_estimators= 50)
for times in range(200):
    df_train, df_test,train_labels,test_labels = train_test_split(data_sel, data_labels, train_size = 0.8, shuffle=True)
    #df_train,train_labels = smote.fit_resample(df_train,train_labels)
    
    
    
    
    forest.fit(df_train,train_labels)
    f_probs = forest.predict_proba(df_test)[:, 1]
    forest_auc = roc_auc_score(test_labels,f_probs)
    stat_forest.append(forest.score(df_test, test_labels))
    
    AUC_forest.append(forest_auc)
print("  Mean AUC ", round(stats.mean(AUC_forest),3),"  Mean ACC ", round(stats.mean(stat_forest),3)) 
    


# In[65]:


stat_forest = []

AUC_forest = []
forest = RandomForestClassifier(random_state=1, criterion= 'gini', max_features= 2, n_estimators= 50)
for times in range(200):
    df_train, df_test,train_labels,test_labels = train_test_split(data_sel2, data_labels, train_size = 0.8, shuffle=True)
    df_train,train_labels = smote.fit_resample(df_train,train_labels)
    
    
    
    
    forest.fit(df_train,train_labels)
    f_probs = forest.predict_proba(df_test)[:, 1]
    forest_auc = roc_auc_score(test_labels,f_probs)
    stat_forest.append(forest.score(df_test, test_labels))
    
    AUC_forest.append(forest_auc)
print("  Mean AUC ", round(stats.mean(AUC_forest),3),"  Mean ACC ", round(stats.mean(stat_forest),3)) 


# In[66]:


stat_forest = []

AUC_forest = []
forest = RandomForestClassifier(random_state=1, criterion= 'gini', max_features= 2, n_estimators= 50)
for times in range(200):
    df_train, df_test,train_labels,test_labels = train_test_split(data_sel3, data_labels, train_size = 0.8, shuffle=True)
    #df_train,train_labels = smote.fit_resample(df_train,train_labels)
    
    
    
    
    forest.fit(df_train,train_labels)
    f_probs = forest.predict_proba(df_test)[:, 1]
    forest_auc = roc_auc_score(test_labels,f_probs)
    stat_forest.append(forest.score(df_test, test_labels))
    
    AUC_forest.append(forest_auc)
print("  Mean AUC ", round(stats.mean(AUC_forest),3),"  Mean ACC ", round(stats.mean(stat_forest),3)) 


# In[69]:




Model_Table = PrettyTable()
Model_Table.field_names = [" ", "   Random forest Classififier  ", "     K-Nearest Neighbours       ", "     XGBoost       "]
Model_Table.add_row(["  Max  ", round(max(stat_forest),3), round(max(stat_knn),3), round(max(stat_XGB),3)])
Model_Table.add_row(["  Min  ", round(min(stat_forest),3),round(min(stat_knn),3),round(min(stat_XGB),3)])
Model_Table.add_row(["  Mean  ", round(stats.mean(stat_forest),3),round(stats.mean(stat_knn),3),round(stats.mean(stat_XGB),3)])
Model_Table.add_row(["  StDev  ", round(stats.stdev(stat_forest),3),round(stats.stdev(stat_knn),3),round(stats.stdev(stat_XGB),3)])


print("Detailed accuracy performance of all models:")
print(Model_Table)#


# In[ ]:




