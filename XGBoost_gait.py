#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#LABIA
source1 = '../data/Gait_Data___Arm_swing.csv'
source2 = '../data/Consensus_Committee_Analytic_Datasets_28OCT21.xlsx'

#Local jupyter
#source2 = "../Quick_Start/Consensus_Committee_Analytic_Datasets_28OCT21.xlsx"
#source1 = "../Motor___MDS-UPDRS/Gait_Data___Arm_swing.csv"


# import data for a first visualisation
data = pd.read_csv(source1)
data.head()

# df_first = merge of all csv sheets from Analytic Datasets
file = source2

df1 = pd.read_excel(file,'PD')
df2 = pd.read_excel(file, 'HC')
df3 = pd.read_excel(file, 'Prodromal')
df4 = pd.read_excel(file, 'SWEDD')

df = pd.concat([df1, df2, df3, df4])

df_first = df.sort_values(['PATNO'])
df_first = df_first.set_index('PATNO')

#df_arm file analyzed
file_arm = source1
df_arm = pd.read_csv(file_arm)

# Get the PATNO of the unknown cohort
df_na_arm = df_arm.loc[df_arm['COHORT'].isna(),['PATNO']]
# Get their cohort
df_complete = df_first.loc[df_na_arm['PATNO'],['Cohort']].replace(['Prodromal',"Parkinson's Disease"],[1,3])
#Replace Nan in df_arm['COHORT']
df_arm.loc[df_arm.COHORT.isnull(), 'COHORT'] = list(df_complete['Cohort'])

# remplace v8 et V8 par V08 idem V2 v6 V10 v10 v11
df_arm["EVENT_ID"].iloc[np.where(df_arm["EVENT_ID"]=='v8')]='V08'
df_arm["EVENT_ID"].iloc[np.where(df_arm["EVENT_ID"]=='V8')]='V08'
df_arm["EVENT_ID"].iloc[np.where(df_arm["EVENT_ID"]=='V2')]='V02'
df_arm["EVENT_ID"].iloc[np.where(df_arm["EVENT_ID"]=='v6')]='V06'
df_arm["EVENT_ID"].iloc[np.where(df_arm["EVENT_ID"]=='v10')]='V10'
df_arm["EVENT_ID"].iloc[np.where(df_arm["EVENT_ID"]=='v11')]='V11'


# In[ ]:


# 1ère méthode pour combler Nan avec KNN

from sklearn.impute import KNNImputer 

imputer = KNNImputer(n_neighbors=3)
df_arm_full = imputer.fit_transform(df_arm.iloc[:,4:])
df_arm_full = pd.DataFrame(df_arm_full, columns=df_arm.iloc[:,4:].columns)
df_arm_full.head(40)


# ## <p style='font-size:30px;text-align:center;color:Cyan'>XGBoost<p>
# <center><img src="images/xgboost.jpg"></center>
# 
# ---
# 
# TODO : 
#     
# 0. faire random forest avec pipeline standardscaler
# 1. faire grille et evaluation des meilleurs hyperparamètres
# 2. voir si avec imputation KNN ou sans mieux ?
# 3. Bons hyperparamètres par defaut pour odele de base??

# #### <p style='font-size:20px;text-align:center;color:DarkBlue'>Recherche des meilleurs Hyperparamètres<p>
# <center><img src='images/xgboost_par.jpg'></center>

# ---
# #### Première approche avec initialisation basique qui sert de référence
# 
# ---

# In[10]:


# Accuracy de la prédiction
from sklearn import metrics
def comp(model, x_test, y_test, title):
    pred = model.predict(x_test)
    print(title+ ' :')
    # Accuracy
    print("Accuracy : ", metrics.accuracy_score(y_test, pred))
    # Matrice de confusion
    conf = metrics.confusion_matrix(y_test, pred)
    normalisation = conf/conf.sum(axis=1)[:, np.newaxis]
    sns.heatmap(normalisation, annot=True, cmap='vlag')
    plt.xticks(np.arange(2)+0.5, ['Parkisons Disease', 'Podromal'], rotation=25)
    plt.yticks(np.arange(2)+0.5, ['Parkisons Disease', 'Podromal'], rotation=0)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Matrice de confusion : ')
    plt.show()
    # Rapport
    print("Rapport :\n{}".format(metrics.classification_report(y_test, pred)))


# In[8]:


import xgboost as xgb
from sklearn import metrics
from pprint import pprint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardSca
from sklearn import model_selection as ms
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

x_train, x_test, y_train, y_test = train_test_split(df_arm_full, df_arm['COHORT'], test_size=0.2)
xgbc = xgb.XGBClassifier()
pipeline = Pipeline([('scaler', StandardScaler()), ('classifieur', xgbc)])
xgbc.fit(x_train, y_train)
comp(xgbc, x_test, y_test, 'Base XGBC')


# In[11]:



# kf = ms.KFold(n_splits=5, shuffle=True)
#kf_scores = ms.cross_val_score(xgbc, x_train, y_train, cv=kf)
#print("K-fold Cross Validation vaut en moyenne: {:.2f}".format(kf_scores.mean()))


# ---
# **Analyse plus fine des paramètres**
# 
# ---

# In[14]:


# 1er balayage sur large éventail de sets d'hyperparamètres avec RandomizedSearchCV

grille_xgb = { 
        'learning_rate': [0.05, 0.1, 0.3, 0.7],
        'n_estimators':[50, 200, 500, 1000],
        'max_depth': [i for i in range(3,20,2)],
        'lambda': [1],
        'min_child_weight': [1, 5, 10],
        'gamma': [0.01, 0.1, 0.5, 1],
        'subsample': [0.1, 0.5, 1, 5],
        'colsample_bytree': [0.05, 0.1, 0.5, 1],
        'max_depth': [2, 5, 10, 20]}

# Choix aléatoire parmi 4*4*9*1*3*4*4*4*4=110592 combinaisons de paramètres possibles


# In[15]:


# Random Search CV training

# 500 combinaisons et K = 5 folds
xgbc_random = RandomizedSearchCV(estimator = xgbc, param_distributions = grille_xgb, n_iter = 500, cv = 5, n_jobs = -1)
xgbc_random.fit(x_train, y_train)
xgbc_random.best_params_


# 500 combinaisons et K=5folds:
# {'subsample': 1,
#  'n_estimators': 1000,
#  'min_child_weight': 1,
#  'max_depth': 2,
#  'learning_rate': 0.7,
#  'lambda': 1,
#  'gamma': 1,
#  'colsample_bytree': 0.1}

# In[ ]:




