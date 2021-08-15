import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#######################  

# Reading the dataset from csv file
print("\nLoading the data set\n")
df = pd.read_csv("./forest_fires_dataset V1.03.csv")
print(df)

# Showing the datatypes
print("\n",df.dtypes)

# Checking for outliers
skewness_hist = df.hist(figsize=(30,10), layout=(2, 7), bins = 15, )

# Pairwise scatter plot
sns.pairplot(df, vars = list(df.columns), kind='scatter', diag_kind='hist')
plt.figure(figsize=(20, 20))
plt.show()



#######################      

## Removing Records with Missing Fields ##
print(df)
print("\nRemoving records with missing fields\n")
df.dropna(inplace=True)
print(df)


## Removing Duplicate Records ##
print("\nRemoving duplicate records\n")
df.drop_duplicates(subset=['temp', 'humidity', 'wind', 'rain'], keep='first', inplace=True)
print(df)


## Removing outliers ##

# converting values to z-score

z = np.abs(stats.zscore(df))
print(z)

# Removing outliers that are more than 3 standard deviations away from mean
print(df.shape)
print("\nRemoving outliers\n")
df = df[(z < 3).all(axis=1)]
print(df.shape)

skewness_hist = df.hist(figsize=(30,10), layout=(2, 7), bins = 15, )

print("\n",df)



#######################    

## Splitting the Data ##

# split the dataset into features data and labels
data = df.drop(['burned_area'], axis = 1)
labels = df['burned_area']

# split data and labels into training set and testing set
training_data, test_data, training_labels, test_labels = train_test_split(data, labels, test_size = 0.30, random_state = 42)


## Training and Predicting ##

# Training the multiple linear regression model
reg_model = LinearRegression().fit(training_data, training_labels)

# Predicting the labels for the test data
test_pred = reg_model.predict(test_data)
print("\nTest data predicted\n", test_pred)


## Intercept and Coefficients ##
print("\nIntercept: \n", reg_model.intercept_, "\n\nCoefficients:\n", reg_model.coef_)


## Assessing the Model ##

# predicting the labels the training data
train_pred = reg_model.predict(training_data)
print("\nTraining data predicted\n")

rmse_train = mean_squared_error(training_labels, train_pred)**0.5
rmse_test = mean_squared_error(test_labels, test_pred)**0.5
print("\nRMSE Train:", rmse_train)
print("RMSE Test:", rmse_test)
print("\nR^2 score:", r2_score(test_labels, test_pred))