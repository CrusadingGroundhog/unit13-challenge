#!/usr/bin/env python
# coding: utf-8

# In[14]:


# Initial imports
import requests
import pandas as pd
import matplotlib.pyplot as plt
import hvplot.pandas
import plotly.express as px
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# In[19]:


#Load the Crypto Data 
## Went with the pre-loaded crypto data. 

crypto_df = pd.read_csv("crypto_data.csv", index_col=0)

crypto_df.head(15)


# ### Data Pre-Processing 
# 
# #### Basically we're trimming down all of the fat on the data. There's a ton of crypto currencies but the relevant ones are far less than the data we have

# In[22]:


# Keep only necessary columns:
# 'CoinName','Algorithm','IsTrading','ProofType','TotalCoinsMined','TotalCoinSupply'
#Keep Crypto that is trading 

crypto_df = crypto_df[crypto_df['IsTrading'] == True]
crypto_df


# In[23]:


##Double checking it worked with different way to get rid of non-trading crypto
crypto_df.drop(crypto_df[crypto_df['IsTrading'] == False].index, inplace=True)


# In[24]:


crypto_df.shape


# In[25]:


# Remove the "IsTrading" column
crypto_df = crypto_df.drop(columns='IsTrading')
crypto_df


# In[26]:


# Remove rows with at least 1 null value (We can see the rows slowly going down)
crypto_df = crypto_df.dropna()
crypto_df


# In[33]:


# Remove rows with cryptocurrencies having no coins mined
crypto_df = crypto_df[crypto_df["TotalCoinsMined"]> 0]
crypto_df


# In[34]:


# Drop rows where there are 'N/A' text values
crypto_df = crypto_df[crypto_df!='N/A']
crypto_df
## Nothing Dropped, I think I did it right but some concern.


# In[36]:


# Store the 'CoinName'column in its own DataFrame prior to dropping it from crypto_df
coin_names = crypto_df['CoinName']
coin_names


# In[37]:


# Drop the 'CoinName' column since it's not going to be used on the clustering algorithm
crypto_df.drop(columns='CoinName', inplace=True)
crypto_df    
##Down to 4 columns


# In[38]:


# Create dummy variables for text features
dummy = pd.get_dummies(crypto_df, columns=['Algorithm', 'ProofType'])
dummy.head()


# In[39]:


# Standardize data
dummy_scaled = StandardScaler().fit_transform(dummy)
dummy_scaled


# ### Reduce Dimensions with PCA

# In[43]:


# Use PCA to reduce dimensions to 3 principal components
pca = PCA(n_components=3)
crypto_pca = pca.fit_transform(dummy_scaled)
##In retrospect shouldn't have used dummy but sticking with it


# In[45]:


# Create a DataFrame with the principal components data
pca_df = pd.DataFrame(
    data=crypto_pca, columns=["principal component 1", "principal component 2", "principal component 3"],
    index= crypto_df.index
)
pca_df.head()


# ### Clustering Crytocurrencies Using K-Means
# #### Find the Best Value for k Using the Elbow Curve

# In[46]:


inertia = []
k = list(range(1, 11))

# Calculate the inertia for the range of k values
for i in k:
    km = KMeans(n_clusters=i, random_state=0)
    km.fit(pca_df)
    inertia.append(km.inertia_)

# Create the Elbow Curve using hvPlot
elbow_data = {"k": k, "inertia": inertia}
crypto_df_elbow = pd.DataFrame(elbow_data)
crypto_df_elbow.hvplot.line(x="k", y="inertia", xticks=k, title="Elbow Curve")


# In[51]:


# Initialize the K-Means model
model = KMeans(n_clusters=4, random_state=0)

# Fit the model
model.fit(pca_df)

# Predict clusters
predictions = model.predict(pca_df)

# Create a new DataFrame including predicted clusters and cryptocurrencies features
clustered_crypto_df=pd.DataFrame({
    "Algorithm": crypto_df.Algorithm,
    "ProofType": crypto_df.ProofType,
    "TotalCoinsMined": crypto_df.TotalCoinsMined,
    "TotalCoinSupply": crypto_df.TotalCoinSupply,
    "PC1": pca_df["principal component 1"],
    "PC2": pca_df["principal component 2"],
    "PC3": pca_df["principal component 3"],
    "CoinName": coin_names,
    "Class": model.labels_,
    },index= crypto_df.index)
clustered_crypto_df.head(10)


#  ### Visualizing Results
# #### 3D-Scatter with Clusters

# In[53]:


# Create a 3D-Scatter with the PCA data and the clusters
fig = px.scatter_3d(
    clustered_crypto_df,
    x="PC3",
    y="PC2",
    z="PC1",
    color="Class",
    symbol="Class",
    width=1000,
    hover_name="CoinName",
    hover_data=["Algorithm"]
)
fig.update_layout(legend=dict(x=0, y=1))
fig.show()


# #### Table of Tradable Cryptocurrencies

# In[55]:


# Table with tradable cryptos
columns = ['CoinName', 'Algorithm', 'ProofType', 'TotalCoinSupply', 'TotalCoinsMined', 'Class']
crypto_table = clustered_crypto_df.hvplot.table(columns)
crypto_table
#There was an option to just name it "table", but I prefer the longer name when I can, it is slower but keeps things clearer. 


# In[57]:


# Print the total number of tradable cryptocurrencies
print(f"Total number of tradable cryptocurrencies is {coin_names.count()}")


# ### Scatter Plot with Tradable Cryptocurrencies

# In[58]:


# Scale data to create the scatter plot
clustered_crypto_df["TotalCoinSupply"]= clustered_crypto_df["TotalCoinSupply"].astype(float)/100000000
clustered_crypto_df["TotalCoinsMined"]= clustered_crypto_df["TotalCoinsMined"].astype(float)/100000000


# In[60]:


# Plot the scatter with x="TotalCoinsMined" and y="TotalCoinSupply"
clustered_crypto_df.hvplot(
    kind="scatter",
    x= "TotalCoinsMined",
    y= "TotalCoinSupply",
    hover_cols=["CoinName"],
    by= "Class")


# In[ ]:


#Turtle Coin and BitTorrent really throw off the scatterplot

