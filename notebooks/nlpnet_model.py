#%% [markdown]
# # requirements
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from torch.utils.data import DataLoader, Dataset, random_split
import torch
import torch.nn as nn

from sentence_transformers import SentenceTransformer
embeddings_model = SentenceTransformer('sentence-transformers/LaBSE')

#%% [markdown]
# loading the data
pathfile = "../data/train.csv"
df = pd.read_csv(pathfile)
df.head()
df = pd.read_csv('../data/train.csv')
df
# %%
df.info()
# %%
df.describe().T
#%% [markdown]
# # feature selection
# ## objective variable
# ## value counts of the objective variable
df['priceRange'].value_counts(dropna=False)

#%% [markdown]
# ## value counts of the objective variable
df[['priceRange']].describe().T

#%% [markdown]
# ## mapping the objective variable
priceRange_map = {
    i: priceRange 
    for priceRange, i in df['priceRange'].value_counts(dropna=False).sort_values().reset_index().to_dict()['priceRange'].items()
    }
print(priceRange_map)

#%% [markdown]
inverted_priceRange_map = {
    priceRange: i
    for i, priceRange in priceRange_map.items()
}
print(inverted_priceRange_map)

#%% [markdown]
# ## transforming the objective variable
df['priceRange'] = df['priceRange'].map(priceRange_map)
df[['priceRange']].describe().T

#%% [markdown]
# ## value counts of the objective variable
df['priceRange'].value_counts(dropna=False)

# %% [markdown]
# ## text variable
df[['description']]

# %% [markdown]
# ## one sample
df['description'][1000]

# %% [markdown]
# ## creating sentence embeddings
class hf_embeddings():
    def __init__(self, model):
        self.model = model

    def __call__(self, texts):
        return self.model.encode(texts)
    
embeder = hf_embeddings(embeddings_model)
samples_sentences = embeder(
        [
            "this is a test sentence",
            "this is another test sentence"
        ]
    )
print(samples_sentences.shape)
print(samples_sentences[0][:10])

#%% [markdown]
# ## creating sentence embeddings for the description variable    
df['description_embedings'] = list(embeder(np.array(df['description'].tolist())))
print(df['description_embedings'].shape)
print(df['description_embedings'].iloc[0][:10])
print(df.head())

# %% [markdown]
# ## creating a dataset class
class csvDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['description_embedings']
        label = self.data.iloc[idx]['priceRange']
        return text, label

dataset = csvDataset(df)
print(f"Dataset length: {len(dataset)}")
print(f"Sample data (text, label): {dataset[0]}")
#%% [markdown]
# ## splitting the dataset into training, validation and test sets
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
print(f"Train dataset length: {len(train_dataset)}")
print(f"Validation dataset length: {len(val_dataset)}")
print(f"Test dataset length: {len(test_dataset)}")

train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True)
val_loader = DataLoader(val_dataset,batch_size=32,shuffle=False)
test_loader = DataLoader(test_dataset,batch_size=32,shuffle=False)

#%% [markdown]
# #modeling
class nn_model(nn.Module):
    def __init__(
            self,
            input_size,
            output_size
    ):
        super(nn_model, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
#%% [markdown]
# ## creating a model instance
input_size = df['description_embedings'].iloc[0].shape[0]
output_size = df['priceRange'].nunique()
model = nn_model(input_size, output_size)
print(model)

#%% [markdown]
# ## defining a loss function and an optimizer for clasification unique labels
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)
num_epochs = 100
#%% [markdown]
# ## training the model
train_losses = []
for epoch in range(num_epochs):
    epoch_train_loss = 0
    for i, (texts, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(texts.float())
        loss = criterion(outputs, labels)
        epoch_train_loss += loss.item()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    train_losses.append(epoch_train_loss / len(train_loader))

#%% [markdown]
# ## evaluating the model on the validation set
val_losses = []
with torch.no_grad():
    correct = 0
    total = 0
    for epoch in range(num_epochs):
        epoch_val_loss = 0
        for texts, labels in val_loader:
            outputs = model(texts.float())
            loss = criterion(outputs, labels)
            epoch_val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        val_losses.append(epoch_val_loss / len(val_loader))
print(f'Validation Accuracy: {100 * correct / total:.2f}%')


#%% [markdown]
# ## comparing the losses of the training and validation sets
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Validation Loss')
plt.legend()
plt.show()


#%% [markdown]
# # saving the model
timestamp = int(datetime.now().timestamp())
print(f"Saving model at timestamp: {timestamp}")
model_path = f'../data/nlpnet_model_{timestamp}.pth'
torch.save(model, model_path)

#%% [markdown]
# # loading trained model class for inference
original_model_path = "../data/nlpnet_model_1778220700.pth"
loaded_model = torch.load(
    original_model_path,
    weights_only=False
)
loaded_model.eval()

#%% [markdown]
# ## making predictions with the loaded model and with test data
with torch.no_grad():
    correct = 0
    total = 0
    for texts, labels in test_loader:
        outputs = loaded_model(texts.float())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Accuracy of the loaded model on the {total} test samples: {100 * correct / total:.2f}%')
