#!/usr/bin/env python
# coding: utf-8

# <br> 
# <body>
# <p style='font-size:50px;text-align:center;color:#0000C6'> Analyses of the GaitPhase Database PPMI<p> 
# <center><img src="images/phases_gait.jpg"></center>
# <p style='font-size:15px;text-align:center'>Decomposition of the different phase of walking</p>
# </body>

# <br>
# <p style='font-size:30px; color:Red'> TODOLIST </p>
# 
# ---
# 
# - Régler pb cohort = 3 voir a quoi correspondes 1/3 dans fichier

# ---
# <p style='font-
# 
# ---
# **Data** : Motor features extracted from the raw accelerometer and gyroscope signals are presented in the tables below.
# <img src = 'image/features.jpg'/>
# 
# **Authors** : Anat Mirelman, PhD, Tel Aviv Medcial Center, Tel Aviv University
# 
# **Data PPMI access** : https://ida.loni.usc.edu/pages/access/studyData.jsp?categoryId=3&subCategoryId=4
# 
# ---
# <font color='red'>I WILL WORK ON IT HERE</font>
# 
# Les données issues des évaluations de la marche dans chaque site sont enregistrées sur un ordinateur désigné sur le site et sont ensuite transférées dans une base de données centrale de la TASMC (Tel Aviv Sourasky Medical Center) pour être traitées. 
# 
# Une fois reçues, l'ingénieur de TASMC examine les données pour en vérifier la qualité et l'intégrité et les traite pour extraire les caractéristiques des signaux bruts de l'accéléromètre et du gyroscope à l'aide d'algorithmes validés. Les données sont ensuite vérifiées à nouveau pour s'assurer de leur qualité et de leur précision.
# 
# L'équipe du TASMC a travaillé avec le CTCC pour créer une légende de données et une infrastructure pour le transfert des données sur le site Web du PPMI (LONI) en tant que données en libre accès (voir le tableau ci-dessous). 
# 
# Tous les sujets conservent leur numéro d'identification unique PPMI, ce qui permet de combiner les données sur la démarche avec d'autres informations recueillies dans le cadre de PPMI.
# 
# ---

# <p style='font-size:30px;text-align:center;color:#00CCDC'> Data Cleaning<p>

# In[1]:


import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# In[2]:


# import data for a first visualisation
data = pd.read_csv('../Motor___MDS-UPDRS/Gait_Data___Arm_swing.csv')
data.head()


# <p style='font-size:15px;color:green'>MERGE CSV : FOCUS ON CLASSIFICATION PD AND HC</p>

# In[3]:


# df_first = merge of all csv sheets from Analytic Datasets
file = '../Quick_Start/Consensus_Committee_Analytic_Datasets_28OCT21.xlsx'

df1 = pd.read_excel(file,'PD')
df2 = pd.read_excel(file, 'HC')
df3 = pd.read_excel(file, 'Prodromal')
df4 = pd.read_excel(file, 'SWEDD')

df = pd.concat([df1, df2, df3, df4])

df_first = df.sort_values(['PATNO'])
df_first = df_first.set_index('PATNO')

#df_arm file analyzed
file_arm = "../Motor___MDS-UPDRS/Gait_Data___Arm_swing.csv"
df_arm = pd.read_csv(file_arm)


df_first.head(10)


# In[19]:


# Search
df_first.loc[40621,['Cohort','Subgroup','Comments']]


# In[4]:


# Display multiple table from stack overflow

from IPython.display import display_html
from itertools import chain,cycle
def display_side_by_side(*args,titles=cycle([''])):
    html_str=''
    for df,title in zip(args, chain(titles,cycle(['</br>'])) ):
        html_str+='<th style="text-align:center"><td style="vertical-align:top">'
        html_str+=f'<h2>{title}</h2>'
        html_str+=df.to_html().replace('table','table style="display:inline"')
        html_str+='</td></th>'
    display_html(html_str,raw=True)
    


# ---
# #### Visualization of number in each cohort in total PPMI base

# In[21]:


# Visualization of number in each cohort in total PPMI base
names = np.unique(df_first['Cohort'])
size = np.divide([np.count_nonzero(df_first['Cohort']=='Healthy Control'),
                  np.count_nonzero(df_first['Cohort'] == "Parkinson's Disease"),
                 np.count_nonzero(df_first['Cohort'] == "Prodromal"),
                 np.count_nonzero(df_first['Cohort'] == "SWEDD")], np.shape(df_first)[0])

# See pie chart
fig, ax = plt.subplots()
explode = (0.1, 0.1, 0.1, 0.1)
ax.pie(size, explode=explode, labels=names, autopct='%1.1f%%', startangle=90)
ax.axis('equal')
ax.set_title('Proportion of PD/Prodromal/HC/SWEDD', color = 'Black')
plt.tight_layout
plt.show()


# ---
# #### A quoi correpsondent n°1/3 de cohort dans fichier ?

# In[6]:


# Visualization of number in cohort=1 in arm swing analyse base

# Filter
cohort_mask = (df_arm.COHORT==1)
col_mask = ['PATNO', 'COHORT']
df_armsort = df_arm.loc[cohort_mask,col_mask]

# Delete duplicates
#df_armsort.drop_duplicates(subset=['PATNO'], keep='first', inplace =True)

# Search corresponding cohort annotation
pat_mask = list(df_armsort.PATNO)
df_corresp = df_first.loc[pat_mask,['Cohort']]


# In[7]:


# Visualization of number in cohort = 3 in arm swing analyse base

# Filter
cohort_mask = (df_arm.COHORT==3)
col_mask = ['PATNO', 'COHORT']
df_armsort3 = df_arm.loc[cohort_mask, col_mask]

# Delete duplicates
#df_armsort3.drop_duplicates(subset=['PATNO'], keep='first', inplace =True)

# Search corresponding cohort annotation
pat_mask = list(df_armsort3.PATNO)
for i in pat_mask:
    try:
        print(df_first.loc[i, ['Cohort']])
    except:
        print('Pas de numéro {} dans la base de donnée'.format(i))

# Error no number 42035?! even on the site not found
pat_mask.remove(42035)
df_corresp3 = df_first.loc[pat_mask, ['Cohort']]
df_armsort3 = df_armsort3.set_index('PATNO')


# In[24]:


# Display
print(df_armsort.shape, df_armsort3.shape)
display_side_by_side(df_armsort, df_corresp, df_armsort3, df_corresp3, titles=['Arm cohort1', 'Corresponding disease', 'Arm cohort3', 'Corresponding disease'])


# ---
# #### <font color='DarkOrange'>Conclusion</font>
# 
# The **cohort 1** on the document corresponds to people in the **Prodromal stage** of Parkinson's disease (the stage at which individuals do not fulfill diagnostic criteria for PD ie, bradykinesia and at least 1 other motor sign)
# 
# The **cohort 3** to patients with **PD**
# 
# *PS : The exceptions of miss-classification are justified in the csv with a column named 'Comments'*
# 
# ---

# ##### Attribution of Nan to the right cohort
# 

# In[6]:


# Get the PATNO of the unknown cohort
df_na_arm = df_arm.loc[df_arm['COHORT'].isna(),['PATNO']]
# Get their cohort
df_complete = df_first.loc[df_na_arm['PATNO'],['Cohort']].replace(['Prodromal',"Parkinson's Disease"],[1,3])
#Replace Nan in df_arm['COHORT']
df_arm.loc[df_arm.COHORT.isnull(), 'COHORT'] = list(df_complete['Cohort'])

df_arm.head(50)


# In[7]:


# remplace v8 et V8 par V08 idem V2 v6 V10 v10 v11
df_arm["EVENT_ID"].iloc[np.where(df_arm["EVENT_ID"]=='v8')]='V08'
df_arm["EVENT_ID"].iloc[np.where(df_arm["EVENT_ID"]=='V8')]='V08'
df_arm["EVENT_ID"].iloc[np.where(df_arm["EVENT_ID"]=='V2')]='V02'
df_arm["EVENT_ID"].iloc[np.where(df_arm["EVENT_ID"]=='v6')]='V06'
df_arm["EVENT_ID"].iloc[np.where(df_arm["EVENT_ID"]=='v10')]='V10'
df_arm["EVENT_ID"].iloc[np.where(df_arm["EVENT_ID"]=='v11')]='V11'


# ---
# #### <p style='font-size:25px;text-align:center;color:Blue'>Quelques statistiques :<p>

# In[27]:


# Statistics of data

print("Nombre de personnes testées : {} et récupération de {} mesures".format(df_arm.shape[0], df_arm.shape[1]-4))
print("Nombre de Prodromal : {}".format(np.shape(np.where(df_arm['COHORT']==1))))
print("Nombre de personnes ayant Parkinson : {}".format(np.shape(np.where(df_arm['COHORT']==3))))


# Visualization of number in each cohort in total PPMI base
sns.countplot(data=df_arm, x="EVENT_ID", hue='COHORT', order=['BL', 'V02', 'V04', 'V06', 'V08', 'V09', 'V10', 'V11', 'V12'])
plt.show()


# In[28]:


df_arm.shape


# In[8]:


# Nan
columns = df_arm.columns
for i, x in enumerate(df_arm[columns].isnull().sum()):
    print('Number of Nan values pour {} : {}   soit {}% de la colonne'.format(columns[i], x, x*100/192))
    
print('Moyenne de {} Nan par colonne soit {}% '.format(np.mean(df_arm[columns].isnull().sum()), round(np.mean(df_arm[columns].isnull().sum())*100/192,2)))


# ### <p style='font-size:25px;text-align:center;color:Blue'> Visualization of data<p> 

# In[12]:


# Several correlations between the measures

corr_pearson = df_arm[4:60].corr(method='pearson')[['COHORT']].sort_values(by='COHORT', ascending=False)
corr_spearman = df_arm[4:60].corr(method='spearman')[['COHORT']].sort_values(by='COHORT', ascending=False)
fig, ax = plt.subplots(1,2, figsize=(10,15))
heatmap = sns.heatmap(corr_pearson, ax=ax[0], vmin=-1, vmax=1, annot=True, cmap='vlag')
heatmap = sns.heatmap(corr_spearman, ax=ax[1],vmin=-1, vmax=1, annot=True, cmap='vlag')
ax[0].set_title('Features Correlating with Cohort : Pearson', fontdict={'fontsize':12}, pad=16);
ax[1].set_title('Features Correlating with Cohort : Spearman', fontdict={'fontsize':12}, pad=16);
plt.tight_layout()
plt.show()


# In[37]:


# Let see the differences
ecart = pd.Series(dtype='float64', index = list(corr_pearson.index), name='Ecart')
for x in list(corr_pearson.index):
    ecart.at[x] = corr_pearson.at[x,'COHORT'] - corr_spearman.at[x,'COHORT']
ecart = ecart.abs()
print(ecart.sort_values())
abscisse = [x if ecart[x]>=0.08 else i  for i,x in enumerate(ecart.index)]

# Print les différences avec val abs de ecart
fig = plt.figure(figsize=(30,12))
plt.plot(abscisse, ecart, marker='+')
plt.tight_layout()
plt.title('Difference entre corr_spearman et corr_pearson pour les mesures')
plt.show()


#  ---
#  #### Analyse
#  
# Pas de grosses différences de correlation entre les colonnes car max de differences de correlation de 0.175 pour TUG2_STRAIGHT_DUR
# 
#  ---

# ---
# ### <font color='green'> Méthodes imputation données manquantes </font>
# 
# ---
# 
# L’imputation de données manquante réfère au fait qu’on remplace les valeurs manquantes dans le jeu de données par des valeurs artificielles. Idéalement, ces remplacements ne doivent pas conduire à une altération sensible de la distribution et la composition du jeu de données.
# 
# ---
# 
# <font color='red'> **Type de missing value :**</font> MCAR/MAR/MNAR et univarié/monotone/arbitraire
# 
# ---
# 
# <font color='red'>**Méthodes :** </font>
# 
# 1. Analyse sans complétion :
#       - Suppresion de données (éviter perte de données)
#       - Méthodes tolérant les données manquantes comme CART/NIPALS/XGBoost (plus complexe et performante : arbre)
# 
# 
# 2. Imputation :
#     - par règle
#     - par mmoyenne ou mode (sensible aux valeurs abbérantes)
#     - Hot Deck(val tirée au hasard parmi val existantes)/ par dernière valeur connu
#     - KNN /Régression locale (LOESS) 
#     - SVD (si bcp plus de données observées que manquantes)/ Complétion SVD (sinon) 
#     - MissForest
#     - Inférence bayésienne/Imputation multiple
#     - Amelia II (mix algo EM et bootstrap (chaque tirage données estimées par bootstrap pour simuler incertitude puis algo EM pour trouver estimateur a posteriori (max de VRSBL)
#     - LOCF missing value
# 

# #### Création d'un fichier exploitable pour Classif
# 

# In[ ]:





# 
# 
#     model = xgb.XGBRegressor()
#     model.fit(X_train, y_train)
#     print(); print(model)
# #Now we have predicted the output by passing X_test and also stored real target in expected_y.
# 
#     expected_y  = y_test
#     predicted_y = model.predict(X_test)
# #Here we have printed r2 score and mean squared log error for the Regressor.
# 
#     print(metrics.r2_score(expected_y, predicted_y))
#     print(metrics.mean_squared_log_error(expected_y, predicted_y))
# 
#     plt.figure(figsize=(10,10))
#     sns.regplot(expected_y, predicted_y, fit_reg=True, scatter_kws={"s": 100})
# 

# **Remarques:**
# - *Analyse :*           
#   Pour lr = 0.01 et max_depth=100 et num_iter=5000 classification acceptable avec accuracy de en moyenne 71%  
# 

# In[38]:


# First vizualisation of speed base walking and dual task walking
fig, ax = plt.subplots(6, 3, figsize = (20,30))
columns = data.columns
for i,ax in enumerate(ax.flat):
    #Columns i+4 pour commencer à plot SP_U
    ax.plot(np.arange(len(data['SP_U'])), np.array(data[columns[i+4]]), label = columns[i+4])
    #Columns i+22 pour commencer à plot SP_U_DT
    ax.plot(np.arange(len(data['SP_U'])), np.array(data[columns[i+22]]), label = columns[i+22])    
    ax.set_title('Comparison {}/{}'.format(columns[i+4],columns[i+22]))
    ax.legend(loc='upper right')
    ax.set_xlabel('Patients')
fig.tight_layout()
plt.show()


# ## <font color='#0000FF'> First classification method</font>

# https://ieeexplore.ieee.org/document/9194627
# 
# R. Arefi Shirvan.et Al ont suggéré d'utiliser le classificateur des voisins les plus proches pour les données de la maladie de Parkinson. Cette technique est utilisée lorsqu'il existe un nombre réduit de caractéristiques qui ne suffisent pas à prédire la variable cible. Cette technique consiste à déterminer les k plus proches voisins et à les regrouper, créant ainsi des clusters. Ce processus a produit une précision de 93 % dans les prédictions, ce qui est remarquable dans la mesure où les données étaient insuffisantes. Dans leur cas, l'ensemble de données comprenait des données vocales provenant de 192 enregistrements effectués par 32 personnes. Les données vocales ont été utilisées comme mesure pour déterminer si le patient présentait ou non les symptômes de la maladie.
# 
# Le Dr R. Geetha Ramani et ses collaborateurs ont proposé une méthode de classification utilisant la régression logistique, l'analyse discriminante linéaire (LDA), les machines à vecteur de support (SVM) et le classificateur de forêt aléatoire. L'ensemble de données a été pris à partir du dépôt d'apprentissage automatique de l'UCI. Il se composait de 22 caractéristiques avec les enregistrements de 197 patients. L'algorithme de l'arbre aléatoire a atteint une précision de 100 %, tandis que le knn et le LDA ont atteint une précision supérieure à 90 %.
# 
# Arvind Kumar Tiwari a proposé dans son article une méthode utilisant un classificateur de forêt aléatoire sur un ensemble de données avec 20 caractéristiques. La précision obtenue en utilisant l'algorithme de sélection des caractéristiques était de 90,3 % avec une précision de 90,2 %. Le coefficient de corrélation était de 0,73 et la valeur roc obtenue était de 0,96. Cette méthode s'est avérée être la meilleure parmi d'autres en termes de précision de prédiction, comme le SVM, le perceptron multicouche et la méthode de l'arbre de décision.
# 
# Anchana Khemphila et al dans son article a utilisé le perceptron multicouche avec l'algorithme de propagation arrière afin de prédire les classes pour la maladie de Parkinson. L'ensemble de données sur la maladie de Parkinson provenant de l'uci machine learning repository a été utilisé. Le gain d'information a été utilisé pour filtrer les caractéristiques au lieu des données du patient. La précision de l'ensemble de données d'apprentissage était de 91,5 %, tandis que la précision de l'ensemble de données de validation était de 80 %.
# 
# Resul Das. Et Al ont utilisé quatre modèles de classification différents pour prédire la maladie de Parkinson. Les modèles utilisés étaient la régression, les réseaux neuronaux, l'arbre de décision et le Dmneural. Les performances de tous ces algorithmes ont été notées. Les réseaux neuronaux ont donné la meilleure précision. Les données de formation étaient de 65% et le reste a été donné pour le test. La précision obtenue par le réseau neuronal était de 92,9 %. L'algorithme BPNN a été utilisé dans le réseau neuronal avec une propagation directe unique et une seule couche cachée dans le réseau.
# 
# Mohammad S Islam.Et Al a utilisé différents classificateurs dans la prédiction. SVM, ANN feed-forward, algorithme d'arbre aléatoire ont été utilisés, et leurs résultats ont été analysés. Les données étaient constituées de plus de 190 échantillons de voix provenant de 30 patients différents. Il s'est avéré que l'algorithme ANN a donné une précision maximale de 97,5 %, tandis que les deux autres algorithmes ont donné une précision insuffisante qui ne convient pas aux applications de soins de santé.
# 

# In[39]:


# Nombre de Nan pour chaque patient pour voir si grosses abérations pour certains patients
graph = []
threshold = []
for i,x in enumerate(list(df_arm.isna().sum(axis=1))):
    graph.append(x)
    if x>=10:
        threshold.append('Nan>=10')
    else:
        threshold.append('Nan<10')
fig = plt.figure(figsize=(15,10))
plt.plot(np.arange(1,len(graph)+1), graph, marker='+')
plt.xlabel('Personne i')
plt.ylabel('Nombre de Nan')
plt.title('Nombre de Nan par personne')
plt.show()


# In[40]:


from collections import Counter
count = [Counter(threshold).get('Nan>=10'), Counter(threshold).get('Nan<10')]
fig = plt.figure(figsize =(10, 7))
plt.pie(count, labels = ['Nan>=10', 'Nan<10'])
plt.title('')
plt.show()


# In[9]:


# 1ère méthode pour combler Nan avec KNN

from sklearn.impute import KNNImputer 

imputer = KNNImputer(n_neighbors=3)
df_arm_full = imputer.fit_transform(df_arm.iloc[:,4:])
df_arm_full = pd.DataFrame(df_arm_full, columns=df_arm.iloc[:,4:].columns)
df_arm_full.head(40)


# ## <p style='font-size:30px;text-align:center;color:Cyan'>Random Forest<p>
# <center><img src="images/random_forest.jpg"></center>

# #### <p style='font-size:20px;text-align:center;color:DarkBlue'>Travail recherche du meilleur set d'hyperparamètres<p>

# Trouver le meilleur set d'hyperparamètre mais sans faire d'overfitting :
# - <p style='color:green'>CrossValidation<p> 
#     
# Demande en calcul enorme car pour chaque set d'hyperparamètre il faut tester K-Fold CV  =>  Nombre d'execution de l'algo = (nbre_set*K) :  
# - <p style='color:green'> Faire un premier balayage Random généralisé sur les set d'hyperparamètre les plus performant + comparer avec set de base<p>
# - <p style='color:green'> Puis effectuer un recherche plus précise avec une grille sur le set random le plus précis précédent + comparer avec set de base<p>
# - <p style='color:green'> Enfin, tableau récapitulatif des performances<p>
# 

# ---
# **RF avec paramètres de base pour référence**
# 
# ---

# In[42]:


# 1- Random Forest avec sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from pprint import pprint

x_train, x_test, y_train, y_test = train_test_split(df_arm_full, df_arm['COHORT'], test_size=0.2)
classif = RandomForestClassifier(n_estimators = 10)

print('Base Parameters utilisés :')
pprint(classif.get_params())
classif.fit(x_train, y_train)

# Remarque : If None, then nodes are expanded until all leaves are pure or until all leaves contain
# less than min_samples_split samples.


# In[14]:


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


# ---
# **Analyse plus fine des paramètres**
# 
# ---

# In[44]:


# 1er balayage sur large éventail de sets d'hyperparamètres avec RandomizedSearchCV

from sklearn.model_selection import RandomizedSearchCV

# Combinaison des classifications d'ensembles d'entraînement générés aléatoirement
bootstrap = [True, False]

# Profondeur max de l'arbre
max_depth = np.linspace(10, 100, num = 10).tolist()

# Number of features to consider when looking for the best split, sqrt=auto, essayer None aussi?
max_features = ['auto', 'sqrt', 'log2']

#the minimum number of samples required to split an internal node,
min_samples_split = [2, 5, 10, 20]

# min_samples_leaf specifies the minimum number of samples required to be at a leaf node
min_samples_leaf = [1, 2, 4]

# Nombre d'arbres dans forêt aléatoire
n_estimators = np.linspace(10, 200, num = 20, dtype = "int64").tolist()

grille = {'bootstrap':bootstrap,
          'max_depth':max_depth,
          'max_features':max_features,
          'min_samples_leaf':min_samples_leaf,
          'min_samples_split':min_samples_split,
          'n_estimators':n_estimators}

# Choix aléatoire parmi 2*10*3*4*3*20=14400 combinaisons de paramètres possibles


# In[45]:


# Random Search CV training

rfc = RandomForestClassifier()
# 100 combinaisons et K = 5 folds
rfc_random = RandomizedSearchCV(estimator = rfc, param_distributions = grille, n_iter = 100, cv = 5, n_jobs = -1)
rfc_random.fit(x_train, y_train)
rfc_random.best_params_


# ---
# Comparison to the base model
# 
# ---
# 
# |Parameters|Base Model|Random Search Model|
# |:----|:----:|----:|
# |n_estimators |10| 160|
# |min_samples_split |2| 10|
# |min_samples_leaf |1| 2|
# |max_features |auto|sqrt|
# |max_depth |None|50.0|
# |bootstrap |True|True|

# In[86]:


# Comparaison
base_rf = classif
best_random_rf = RandomForestClassifier(n_estimators = 150, min_samples_split = 10, min_samples_leaf = 2, max_features = 'sqrt', max_depth = 50, bootstrap = True).fit(x_train, y_train)
comp(base_rf, x_test, y_test, 'Base rf')
comp(best_random_rf, x_test, y_test, 'Meilleur random rf')


# In[48]:


# Plus précis autour des meilleurs paramètres random -> 576 sets

best_grille = {
    'bootstrap' : [True],
    'max_depth' : [40, 50, 60, 70, 80, 90],
    'max_features': ['sqrt', 'log2'], 
    'min_samples_leaf': [1, 2],
    'min_samples_split': [1, 5, 10, 20],
    'n_estimators' : [150, 160, 170, 180, 190, 200]
}


# In[49]:


# Random Search CV training
from sklearn.model_selection import GridSearchCV
rfc = RandomForestClassifier()

# 100 combinaisons et K = 5 folds
rfc_best = GridSearchCV(estimator = rfc, param_grid = best_grille, cv = 5, n_jobs = -1)
rfc_best.fit(x_train, y_train)
rfc_best.best_params_


# In[52]:


base_rf = classif
best_rf = RandomForestClassifier(n_estimators = 160, min_samples_split = 10, min_samples_leaf = 2, max_features = 'sqrt', max_depth = 60, bootstrap = True).fit(x_train, y_train)
comp(best_rf, x_test, y_test, 'Best rf')


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

# In[23]:


import xgboost as xgb
from sklearn import metrics
from pprint import pprint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

x_train, x_test, y_train, y_test = train_test_split(df_arm_full, df_arm['COHORT'], test_size=0.2)
xgbc = xgb.XGBClassifier()
pipeline = Pipeline([('scaler', StandardScaler()), ('classifieur', xgbc)])
xgbc.fit(x_train, y_train)
print('Base Parameters utilisés :')
pprint(xgbc.get_params())


# In[26]:


from sklearn import model_selection as ms
comp(xgbc, x_test, y_test, 'Base XGBC')
kf = ms.KFold(n_splits=5, shuffle=True)
kf_scores = ms.cross_val_score(xgbc, x_train, y_train, cv=kf)
print("K-fold Cross Validation vaut en moyenne: {:.2f}".format(kf_scores.mean()))


# ---
# **Analyse plus fine des paramètres**
# 
# ---

# In[30]:


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


# In[32]:


# Random Search CV training

# 500 combinaisons et K = 5 folds
xgbc_random = RandomizedSearchCV(estimator = xgbc, param_distributions = grille_xgb, n_iter = 50, cv = 5, n_jobs = -1)
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

# **Remarques:**
# - *Analyse :*           
#   Pour lr = 0.01 et max_depth=100 et num_iter=5000 classification moyenne avec accuracy moyenne de 58%  
#   
# - *Solutions :*  
#   Changer la mesure sur laquelle nous effectuons une classification car 0.44 de corrélation,   
#   c'était attendu que la classification ne soit pas bonne / se baser sur plus de mesures ?/ imputation des Nan nécessaire ?

# In[37]:


acc = []
for i in range(15):
    acc.append(xgboost(0.01, 5000, 100, df_arm.iloc[:,4:], df_arm['COHORT']))
print('Mean Accucacy for fixed parameters : {}'.format(np.mean(np.array(acc))))


# In[ ]:




