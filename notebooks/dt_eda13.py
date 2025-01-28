#%%[markdown]
# # requiriments
#%%
# base
import sys
import os
#cleaning
import re
#manupulate numbers
import numpy as np
import pandas as pd
import random
#Graph
import seaborn  as sns
import matplotlib.pyplot as plt
from datetime import date,datetime,timedelta
#store
import pickle
#method
sys.path.append('..')
from src.settings import Settings
settings = Settings()
#modeling
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
#ensamble
from sklearn.tree import DecisionTreeClassifier
#%%[markdown]
# # Data
# %%
df = pd.read_csv('../data/train.csv')
df
# %%
df.info()
# %%
# %%
df_types = pd.DataFrame(df.dtypes)
object_features = df_types[df_types[0] == 'object'].index.to_list()
object_features
# %%
float_features = df_types[df_types[0] == 'float64'].index.to_list()
float_features
# %%
int_features = df_types[df_types[0] == 'int64'].index.to_list()
int_features
# %%
bool_features = df_types[df_types[0] == 'bool'].index.to_list()
bool_features
#%%
df.describe().T
# %%
df[object_features].describe(include='all').T
# %%
for col in object_features:
    print(f'------>{col} : {df[col].unique()}')
    print(f'------>{col} : {df[col].value_counts(dropna=False)}')
# %%
df['priceRange'].unique().tolist()
#%%[markdown]
# ## null values visualization
# %%
sns.heatmap(df.isnull(), cbar=False)
plt.title("data personas")
plt.xlabel('Variable')
plt.ylabel('Fila')
#%%[markdown]
# # numeric variables
# ## variables distribution
# %%
var_hist = int_features + float_features
plt.figure(figsize=(25,6*len(var_hist)/2))
plt.title("Distribución de Variables Numericas")
for i,var in enumerate(var_hist):
    plt.subplot(round(len(var_hist)/2),2,i+1)
    sns.histplot(df, x=var, color='y', kde=True)
    plt.xlabel(var)
    plt.ylabel("count")
# %%
sns.boxplot(x=df['lotSizeSqFt'])
# %%
df['lotSizeSqFt'].describe([0.9,0.95,0.99])
# %%
df['lotSizeSqFt'].value_counts(dropna=False)
# %%
df[df['lotSizeSqFt']>df['lotSizeSqFt'].quantile(0.999)]
# %%
df['log_lotSizeSqFt'] = np.log(df['lotSizeSqFt'])
sns.boxplot(x=df['log_lotSizeSqFt'])
# %%
sns.histplot(df['log_lotSizeSqFt'], kde=True)

# %% [markdown]
# ## categorical variables
object_features = [obj_var for obj_var in object_features if obj_var not in ['description']]
object_features
# %%
city_map = {i: city for city,i in df['city'].value_counts().sort_values().reset_index().to_dict()['city'].items()}
city_map
# %%
homeType_map = {i: homeType for homeType,i in df['homeType'].value_counts().sort_values().reset_index().to_dict()['homeType'].items()}
homeType_map
# %%
priceRange_map = {i: priceRange for priceRange,i in df['priceRange'].value_counts().sort_values().reset_index().to_dict()['priceRange'].items()}
priceRange_map
# %%
maps_dict = {'city': city_map, 'homeType': homeType_map, 'priceRange': priceRange_map}
for col in object_features:
    df[col] = df[col].map(maps_dict[col])

# %%
df[object_features].describe(include='all').T
# %% [markdown]
# ## bool variables
# %%
df[bool_features] = df[bool_features].astype(int)
df[bool_features]
# %%
df[bool_features].describe().T
# %%
df[bool_features].value_counts(dropna=False)
# %% [markdown]
# ## Variables correlation
# %%
float_features = [int_var if 'lotSizeSqFt' not in int_var else 'log_lotSizeSqFt' for int_var in float_features]
correlation_features = int_features + float_features + bool_features + object_features
df_correlations = df[df.columns[df.columns.isin(correlation_features)]].corr()

#mask the upper half for visualization purposes
mask = np.zeros_like(df_correlations, dtype=bool)
mask[np.triu_indices_from(mask)] = True

# Draw the heatmap with the mask and correct aspect ratio
#sns.set(font_scale=1.5)
#plt.style.use("dark_background")
plt.figure(figsize= (10,10))
sns.heatmap(df_correlations, mask=mask, cmap="YlOrBr",#"RdYlBu",
    annot=True, square=True,
    vmin=-1, vmax=1,
    fmt="+.1f")
plt.title("Numeric variables correlations")

# %% [markdown]
# ## feature selection
# %%
df[correlation_features[1:]].describe(include='all').T
# %%
print(f'{len(correlation_features[1:])} features selectionated for the modeling')
# %%
correlation_features[1:]
# %%
df_train = df[correlation_features[1:]].copy()
df_train
# %%
df_train
# %%
correlation_features[1:-1]
# %%
df_train[correlation_features[1:-1]]
# %% [markdown]
# # sampling
# %%
X = np.array(df_train[correlation_features[1:-1]])
y = np.array(df_train[correlation_features[-1]])
print(f'X shape: {X.shape} y shape: {y.shape}')
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=2025)
# %% [markdown]
# # modeling
# %%
dt=DecisionTreeClassifier(
    random_state=2025,
    class_weight='balanced'
)
multilabel_dt = OneVsRestClassifier(dt)
# %%
param_grid={
    'estimator__max_features':  ['sqrt'],
    'estimator__max_depth' : range(5,10,5),
    'estimator__min_samples_leaf': range(5,10,5),
    'estimator__min_samples_split': range(5,10,5),
    'estimator__criterion' :['gini'],
    'estimator__ccp_alpha' : [0]
}
# %%
CV_dtc = GridSearchCV(
    estimator=multilabel_dt,
    param_grid=param_grid,
    scoring='f1_micro',
    cv= 5
)
CV_dtc.fit(X_train, y_train)
# %%
print("Best: %f using %s" % (CV_dtc.best_score_, CV_dtc.best_params_))
# %%
y_hat = CV_dtc.predict(X_test)
print("Accuracy for Decision tree on CV data: ",accuracy_score(y_test,y_hat))
# %%
print("F1 score for Decision tree on CV data: ",f1_score(y_test,y_hat, average='micro'))
# %%
print(classification_report(y_test, y_hat))
# %% [markdown]
# # Save model
# %%
timestamp = int(datetime.utcnow().timestamp())
pathfile = f'../data/{timestamp}_dt_eda13.pkl'
print(f' Stored model: {pathfile}')
pickle.dump(CV_dtc, open(pathfile, "wb"))
# %% [markdown]
# # Validation
## local model
# %%
pkl_filename_local = '../data/1738124306_dt_eda13.pkl'
with open(pkl_filename_local, 'rb') as file:
    uploaded_CV_dtc_local = pickle.load(file)
# %%
y_hat_dtc = uploaded_CV_dtc_local.predict(X_test)
print("Test f1_score: {0:.4f} %".format(100 * f1_score(y_test,y_hat_dtc, average='micro')))
# %%
print("Test f1_score: {0:.4f} %".format(100 * f1_score(y_test,y_hat_dtc, average='macro')))
# %%
print(classification_report(y_test, y_hat_dtc))
