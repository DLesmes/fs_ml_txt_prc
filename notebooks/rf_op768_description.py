#%% [markdown]
# # requirements
# %%
# base
import sys
import os
# cleaning
import numpy as np
import pandas as pd
import random
import re
#graph
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date, datetime, timedelta
# store
import pickle
# embbeds
import torch
import torch.nn.functional as F
import open_clip
# modeling
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score
# ensamble
from sklearn.ensemble import RandomForestClassifier
# %%
MODEL_ID = "hf-hub:timm/ViT-B-16-SigLIP-i18n-256"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, preprocess = open_clip.create_model_from_pretrained(
    MODEL_ID,
    device=device,
    precision='fp16'
)
model.to(device)
model.eval()
tokenizer = open_clip.get_tokenizer(MODEL_ID)
# %%
def get_prompt_embeddings(prompt: str):
    text_input = tokenizer(prompt).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_input)
        text_features = text_features/text_features.norm(dim=1, keepdim=True)
    return text_features
#%% [markdown]
# # data
df = pd.read_csv('../data/train.csv')
df
# %%
df.info()
# %%
df.describe().T
#%% [markdown]
# # feature selection
# ## objective variable
priceRange_map = {i: priceRange for priceRange, i in df['priceRange'].value_counts(dropna=False).sort_values().reset_index().to_dict()['priceRange'].items()}
df['priceRange'] = df['priceRange'].map(priceRange_map)
df[['priceRange']].describe().T
#%% [markdown]
# ## predictive features
len(get_prompt_embeddings(df['description'][0])[0])
#%%
[get_prompt_embeddings(x) for x in df['description'][:10]]
#%%
pathfile = '../data/1738232105_tuple_embbeds.pkl'
with open(pathfile, 'rb') as file:
    uploaded_embbeds = pickle.load(file)
#%%
type(uploaded_embbeds)
# %%
list(uploaded_embbeds.keys())[:10]
# %%
df['uid'][:10]
# %%
list(uploaded_embbeds.values())[:1][-760:]
# %%
df['embbed_features'] = df['uid'].map(uploaded_embbeds)
df['embbed_features']
#%% [markdown]
# # sampling
df.columns
# %%
df_train = df[['priceRange', 'embbed_features']].copy()
df_train
# %%
X = np.array(list(df_train['embbed_features']))
y = np.array(list(df_train['priceRange']))
print(f'X shape: {X.shape}, y shape: {y.shape}')
# %%
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=2025)
# %%
#%% [markdown]
# # modeling
# %%
rf = RandomForestClassifier(
    random_state=2025,
    n_jobs=-1,
    class_weight='balanced'
)
# %%
param_grid = {
    'max_features':['log2'],
    'max_depth': [5],
    'min_samples_leaf': [0.001],
    'min_samples_split': [15],
    'criterion':['gini'],
    'n_estimators':[50]
}
# %%
CV_rfc = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5
)
CV_rfc.fit(X_train, y_train)
#%%
print(f"Best params {CV_rfc.best_params_}")
# %%
print(f"Best score {CV_rfc.best_score_}")
# %%
y_hat = CV_rfc.predict(X_test)
print(f'Accuracy for Random fores on test data is: {accuracy_score(y_test,y_hat)}')
# %%
print(f"F1-score for Random fores on test data is: {f1_score(y_test,y_hat, average='micro')}")
#%%
print(classification_report(y_test,y_hat))
#%% [markdown]
# # saving
timestamp = int(datetime.utcnow().timestamp())
pathfile = f'../data/{timestamp}_rf_op768_description.pkl'
print(f'the model is saved in :{pathfile}')
pickle.dump(CV_rfc, open(pathfile, 'wb'))
#%% [markdown]
# # local validation
#%%
pathfile = '../data/1738236881_rf_op768_description.pkl'
with open(pathfile, 'rb') as file:
    uploaded_cv_rfc = pickle.load(file)
#%%
y_hat = uploaded_cv_rfc.predict(X_test)
print(f'Accuracy for Random fores on test data is: {accuracy_score(y_test,y_hat)}')
# %%
print(f"F1-score for Random fores on test data is: {f1_score(y_test,y_hat, average='micro')}")
#%%
print(classification_report(y_test,y_hat))
