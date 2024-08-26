#!/usr/bin/env python
# coding: utf-8

# In[113]:


# The following code is for training an autoencoder to detect anomalies (fraud cases) in a dataset.
# We will preprocess the data, train the autoencoder on normal data,
# and identify potential fraud cases based on reconstruction error.

# Importing required libraries
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


# In[114]:


# Define onpath and outpath
inpath = pathlib.Path(r"C:\Azar_Drive\Fraud detection\Data")


# In[115]:


# Read data
data_main= pd.read_csv(inpath/"processed_data_part_1.csv", quoting=csv.QUOTE_ALL, low_memory=False)
data_fraud= pd.read_csv(inpath/"ExactMatchNPI_WithState.csv", quoting=csv.QUOTE_ALL, low_memory=False)


# In[117]:


# Add a new column 'Fraud' to the main dataset
data_main['Fraud'] = 0


# In[118]:


# Check if each NPI in the first dataset is present in the second dataset
# Assign 1 to 'fraud' column if NPI is present, else assign 0
data_main['Fraud'] = data_main['Prscrbr_NPI'].apply(lambda x: 1 if x in data_fraud['Prscrbr_NPI'].values else 0)

# Save the updated dataset1 to a new CSV file
data_main.to_csv('data_fraud.csv', index=False)


# In[119]:


# Take a sample containing 200 unlabeled onstances plus the labeled instances
data3 = data_main[data_main["Fraud"] !=1].sample(n=200)
data2 = data_main[data_main["Fraud"] ==1]


# In[121]:


data = pd.concat([data3,data2], ignore_index=True)


# In[124]:


X = data.drop(columns=['Fraud'])  # features
y = data['Fraud']  # target variable


# In[125]:


# Identify numerical and categorical columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns


# In[126]:


# Define preprocessing steps for numerical and categorical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


# In[127]:


# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])


# In[ ]:


# Apply preprocessing
X_preprocessed = preprocessor.fit_transform(X)


# In[128]:


# Filter the data
normal_data = data[data['Fraud'] == 0]
fraud_data = data[data['Fraud'] == 1]


# In[129]:


# Split normal data into training and testing sets
X_normal = normal_data.drop(columns=['Fraud'])
y_normal = normal_data['Fraud']
X_train, X_test, y_train, y_test = train_test_split(X_normal, y_normal, test_size=0.2, random_state=42)


# In[132]:


# Preprocess the filtered normal data
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)


# In[134]:


# Convert training and testing sets to PyTorch tensors
train_tensor = torch.tensor(X_train_preprocessed.toarray(), dtype=torch.float32)
test_tensor = torch.tensor(X_test_preprocessed.toarray(), dtype=torch.float32)


# In[135]:


# Create TensorDatasets and DataLoaders for training and testing
train_dataset = TensorDataset(train_tensor, train_tensor)
test_dataset = TensorDataset(test_tensor, test_tensor)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# In[137]:


# Define the autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(train_tensor.shape[1], 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, train_tensor.shape[1])
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# In[138]:


# Initialize the autoencoder, loss function, and optimizer
autoencoder = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)


# In[139]:


# Training the autoencoder
num_epochs = 50
for epoch in range(num_epochs):
    for data in train_dataloader:
        inputs, _ = data
        # Forward pass
        outputs = autoencoder(inputs)
        loss = criterion(outputs, inputs)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training completed.")


# In[140]:


# Set the autoencoder to evaluation mode
autoencoder.eval()

# Compute reconstruction error for testing set
with torch.no_grad():
    reconstructed = autoencoder(test_tensor)
    reconstruction_error = torch.mean((reconstructed - test_tensor) ** 2, dim=1)


# In[153]:


# Convert the errors to a DataFrame and set an anomaly threshold
error_df = pd.DataFrame(reconstruction_error.numpy(), columns=['Reconstruction_Error'])
threshold = error_df['Reconstruction_Error'].quantile(0.85)  # Example threshold
fraud_cases = error_df[error_df['Reconstruction_Error'] > threshold]

print(f"Detected {len(fraud_cases)} potential fraud cases in the testing set.")


# In[154]:


fraud_cases


# In[157]:


y.iloc[19]


# In[ ]:




