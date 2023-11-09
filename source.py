#!/usr/bin/env python
# coding: utf-8

# In[36]:


#Importing Libraries
import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.model_selection import ShuffleSplit


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor


# In[2]:


#Reading Data from OrderData file

#directoryPath = r"C:\Users\Fatima Aamir\Downloads\AI-Project Dataset\training_data\training_data\order_data"
directoryPath = r"C:\Users\shame\Downloads\AI-Project Dataset\training_data\training_data\order_data"
files = glob.glob(os.path.join(directoryPath, '*'))
orderData = pd.concat((pd.read_csv(f, delimiter="\t", names=["order_ID","driver_id","passenger_id","start_region_hash","dest_region_hash","Price","Time"]) for f in files), ignore_index=True)


# In[3]:


#Display Data
orderData.head()


# In[4]:


#Trimming time to exclude seconds
orderData["Time"] = orderData["Time"].str.slice(stop=16)


# In[5]:


#Display Data
orderData.head()


# In[6]:


#Reading Data from the POI file
#poi = pd.read_csv(r'C:\Users\Fatima Aamir\Downloads\AI-Project Dataset\training_data\training_data\poi_data\poi_data', header=None)
poi = pd.read_csv(r'C:\Users\shame\Downloads\AI-Project Dataset\training_data\training_data\poi_data\poi_data', header=None)
poi[[0,1]] = poi[0].str.split('\t', n=1, expand=True); poi[1] = poi[1].str.replace('\t', '  '); poi.columns = ['RegionHash', 'PoiClass']


# In[7]:


#Display Data
poi.head()


# In[8]:


#Reading Data from WeatherData file

#directoryPath = r"C:\Users\Fatima Aamir\Downloads\AI-Project Dataset\training_data\training_data\weather_data"
directoryPath = r"C:\Users\shame\Downloads\AI-Project Dataset\training_data\training_data\weather_data"

files = glob.glob(os.path.join(directoryPath, '*'))
weatherData = pd.concat((pd.read_csv(f, delimiter="\t", names=["Time","Weather","Temperature","PM2.5"]) for f in files), ignore_index=True)


# In[9]:


#Display Data
weatherData.head()


# In[10]:


#merging the weatherData dataset with the orderData dataset
mergedData = pd.merge(orderData, weatherData, on="Time", how="left")


# In[11]:


#Display Data
mergedData.head()


# In[12]:


#Dropping duplicate rows
mergedData = mergedData.drop_duplicates()


# In[13]:


#Reading Data from clusterData file

#regionData = pd.read_csv(r"C:\Users\Fatima Aamir\Downloads\AI-Project Dataset\training_data\training_data\cluster_map\cluster_map",delimiter="\t",names=["region hash","region id"])
regionData = pd.read_csv(r"C:\Users\shame\Downloads\AI-Project Dataset\training_data\training_data\cluster_map\cluster_map",delimiter="\t",names=["region hash","region id"])


# In[14]:


#Display Data
regionData.head()


# In[15]:


#
regionInformation = dict(zip(regionData["region hash"], regionData["region id"]))
mergedData['start_region_hash'] = mergedData['start_region_hash'].replace(regionInformation)
mergedData['dest_region_hash'] = mergedData['dest_region_hash'].replace(regionInformation)


# In[16]:


#Dropping the first three columns
plot = mergedData.drop(mergedData.columns[:3], axis=1)


# In[ ]:


#Displaying a pairplot of the entire dataframe
sns.pairplot(plot)


# In[18]:


#Display Data
mergedData.head()


# In[19]:


#Filling out the NULL values 
mergedData.fillna(0,inplace=True)


# In[20]:


#Computing the supply and demand 

regionInformation_dict = dict(regionInformation)
orderData["start_region_hash"] = orderData["start_region_hash"].replace(regionInformation_dict)

supplyData = orderData[["driver_id", "start_region_hash", "Time", "dest_region_hash"]].dropna()
supplyDataset = supplyData.groupby(["Time", "start_region_hash"]).size().reset_index(name='Supply')

demandData = orderData[["passenger_id", "start_region_hash", "Time", "dest_region_hash"]].dropna()
demandDataset = demandData.groupby(["Time", "start_region_hash"]).size().reset_index(name='Demand')


# In[21]:


#Display Data
supplyDataset.head()


# In[22]:


#Display Data
demandDataset.head()


# In[23]:


#Joining the two datasets
finalData = demandDataset.set_index(["Time", "start_region_hash"]).join(supplyDataset.set_index(["Time", "start_region_hash"]), how="inner").reset_index()


# In[24]:


#
finalData = pd.merge(finalData, mergedData[["start_region_hash", "Time", "Weather", "Temperature", "PM2.5"]], on=["Time", "start_region_hash"], how="left")
finalData = finalData.drop_duplicates().reset_index(drop=True)


# In[25]:


#Display Data
finalData


# In[26]:


#Computing the Time,Year,Month,Day,Hour,Min Column from the 'Time' Column
finalData["Time"] = pd.to_datetime(finalData["Time"])
finalData["year"] = finalData["Time"].dt.year
finalData["month"] = finalData["Time"].dt.month
finalData["day"] = finalData["Time"].dt.day
finalData["hour"] = finalData["Time"].dt.hour
finalData["min"] = finalData["Time"].dt.minute
finalData = finalData.drop("Time", axis=1)


# In[27]:


#Creating a new Column 'Gap' using Demand and Supply
finalData["Gap"] = finalData["Demand"]-finalData["Supply"]


# In[43]:


sns.scatterplot(data=finalData, x='Gap', y='Demand')


# In[45]:


sns.scatterplot(data=finalData, x='Gap',  y='Supply')


# In[28]:


#Display Data
finalData.head()


# In[29]:


#Display Data
finalData.head(-5)


# In[30]:


#Adding Column names to our dataset
final = finalData[['year','month','day','hour','min','start_region_hash','Weather','Temperature','PM2.5','Gap']]


# In[31]:


#Display Data
final.head(-5)


# In[32]:


#Splitting the data into Testing and Training Data
splitter = ShuffleSplit(n_splits=1, test_size=0.1, random_state=52)
train_idx, test_idx = next(splitter.split(final.drop("Gap", axis=1), final["Gap"]))


# In[33]:


#Splitting the data into X and y arrays
X = final.drop("Gap", axis=1).values
y = final["Gap"].values


# In[34]:


#Using the indices from the splitter to get the train and test sets
x_train, x_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]


# In[37]:


#Using Decision Tree Regression Model
decisionTreeRegressionModel = DecisionTreeRegressor(max_depth=5)
decisionTreeRegressionModel.fit(x_train,y_train)
y_predicted = decisionTreeRegressionModel.predict(x_test)
meanAbsError = mean_absolute_error(y_test, y_predicted)
accuracy = r2_score(y_test, y_predicted) 

print("Mean absolute error:", meanAbsError)
print("Accuracy:", accuracy, "%")

plt.scatter(y_test, y_predicted, s=50, c='red')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values')
plt.show()


# In[38]:


# Using RandomForestRegressor Model
randomForestModel = RandomForestRegressor()
randomForestModel.fit(x_train, y_train)
y_predicted = randomForestModel.predict(x_test)
meanAbsError = mean_absolute_error(y_test, y_predicted)
accuracy = r2_score(y_test, y_predicted)

print("Mean absolute error:", meanAbsError)
print("Accuracy:", accuracy, "%")

plt.scatter(y_test, y_predicted, s=50, c='yellow')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values')
plt.show()


# In[39]:


from sklearn.ensemble import AdaBoostRegressor
reg = AdaBoostRegressor()
reg.fit(x_train,y_train)
y_predicted = reg.predict(x_test)
meanAbsError = mean_absolute_error(y_test, y_predicted)
accuracy = r2_score(y_test, y_predicted)

print("Mean absolute error:", meanAbsError)
print("Accuracy:", accuracy, "%")

plt.scatter(y_test, y_predicted, s=50, c='blue')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values')
plt.show()


# In[46]:


#OPTIMIZATION CODE BELOW


# In[47]:


from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


# In[48]:


decisionTreeRegressor = DecisionTreeRegressor(max_depth=5, min_samples_leaf=2, min_samples_split=2)
decisionTreeRegressor.fit(x_train, y_train)
y_predicted = decisionTreeRegressor.predict(x_test)
meanAbsError = mean_absolute_error(y_test, y_predicted)
print("Mean absolute error:", meanAbsError)


# In[49]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
test_grid = {'max_depth': [3, 5, 7, 9], 
              'min_samples_leaf': [1, 2, 3],
              'min_samples_split': [2, 3, 4]}
regression1 = DecisionTreeRegressor()

searching = GridSearchCV(regression1, test_grid, cv=5, scoring="neg_mean_absolute_error")
searching.fit(x_train, y_train)

print("Best parameters:", searching.best_params_)
print("Best score:", -searching.best_score_)

regression1 = DecisionTreeRegressor(max_depth=searching.best_params_['max_depth'], 
                            min_samples_leaf=searching.best_params_['min_samples_leaf'], 
                            min_samples_split=searching.best_params_['min_samples_split'])

regression1.fit(x_train, y_train)
y_pred = regression1.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)
print("Mean absolute error:", mae)


# In[50]:


# Using RandomForestRegressor Model
randomForestModel = RandomForestRegressor()
randomForestModel.fit(x_train, y_train)
y_predicted = randomForestModel.predict(x_test)
meanAbsError = mean_absolute_error(y_test, y_predicted)
accuracy = r2_score(y_test, y_predicted)

print("Mean absolute error:", meanAbsError)
print("Accuracy:", accuracy, "%")


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

standard_Scaler = StandardScaler()
x_train = standard_Scaler.fit_transform(x_train)
x_test = standard_Scaler.transform(x_test)

test_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
searching = GridSearchCV(RandomForestRegressor(random_state=42), test_grid, cv=5)
searching.fit(x_train, y_train)
forestModel = searching.best_estimator_
forestModel.fit(x_train, y_train)
y_predicted = forestModel.predict(x_test)

mae = mean_absolute_error(y_test, y_predicted)
accuracy = r2_score(y_test, y_predicted)

print("Mean absolute error:", mae)
print("Accuracy:", accuracy)

