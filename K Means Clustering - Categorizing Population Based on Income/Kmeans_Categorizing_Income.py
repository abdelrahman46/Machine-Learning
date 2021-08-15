import csv
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the data

with open('LifeCycleSavings V1.02.csv', 'r') as file:
    reader = csv.reader(file)
    feature_names = [next(reader)]
    features = list(reader)

print("\nLoaded data")
print("Feature Names:\n", feature_names)
print("\nFeatures:\n", features)

### NOISE AND ANOMALY REMOVAL ###

# Cleaning Noise 1: remove records with anomaly of having empty fields

for record in features[:]:
    for value in record[:]:
        if value == '':
            features.remove(record)
            break
            
print("\nCleaned Noise 1: remove records with anomaly of having empty fields\n")
print("\nFeatures:\n", features)     


# Cleaning Noise 2: remove duplicate records

no_duplicates = []

for record in features:
    if record not in no_duplicates:
        no_duplicates.append(record)
        
features = no_duplicates

print("\nCleaned Noise 2: remove duplicate records\n")
print("\nFeatures:\n", features)
     
            
# Transforming feature_names and features to numpy array

feature_names = np.array(feature_names)
features = np.array(features)


# Cleaning Noise 3: remove the irrelevant feature Country 

feature_names = feature_names[:, 1:]
features = features[:, 1:]
print("\nCleaned Noise 3: remove the irrelevant feature Country\n")
print("Feature Names:\n", feature_names)
print("\nFeatures:\n", features)



# Removing outliers

print("\nRemoving Outliers")

# convert the features' values to floats

features_floats = features.astype(np.float)
print(features_floats)

# calculate Q1, Q3, and IQR for each feature
Q1 = []
Q3 = []
IQR = []

for column in features_floats.T:
    sorted_column = np.sort(column)
    q1 = np.quantile(sorted_column,0.25)
    Q1.append(q1)
    q3 = np.quantile(sorted_column,0.75)
    Q3.append(q3)
    IQR.append(q3 - q1)

print("\n1st Quartile:\n", Q1)
print("\n3rd Quartile:\n", Q3)
print("\nInter Quartile Range:\n", IQR)

# calculate upper bound and lower bound for each feature

upper_bound = []
lower_bound = []

for i in range(len(features.T)):
    upper_bound.append(Q3[i]+1.5*IQR[i])
    lower_bound.append(Q1[i]-1.5*IQR[i])
    
print("\nUpper Bound:\n", upper_bound)
print("\nLower Bound:\n", lower_bound)
    
# Remove records where they have values which exceed the bounds

outliers_list = []

# change features to list for convenience

features = features.tolist()

for i in range(len(features)):
    for j in range(len(features[0])):
        if features_floats[i][j] > upper_bound[j] or features_floats[i][j] < lower_bound[j]:
            outliers_list.append(features[i])
            
for outlier in outliers_list:
    features.remove(outlier)

# change features back to numpy array

features = np.array(features)    
            
print("\nOutliers Detected:\n", outliers_list)
print("\nFeatures:\n", features)   
    
    

### CATEGORIZING THE DATA USING K-MEANS CLUSTERING ALGORITHM ###

# convert the features' values to floats

features_floats = features.astype(np.float)
print(features_floats)

# sort features based on income

sorted_features = features_floats[np.argsort(features_floats[:, 3])]
print(sorted_features)

# extract income column

income = sorted_features[:, 3]

# reshape column to use in KMeans function

income = np.reshape(income, (-1,1))
print(income)

# using KMeans function to form 3 clusters from income

income_kmeans = KMeans(n_clusters=3, random_state=0).fit(income)

# get the labels
labels = income_kmeans.labels_
print(labels)

# match category names to labels 
label_names = []
for label in labels:
    if str(label) not in label_names:
        label_names.append(str(label))
category_names = ["low", "medium", "high"]
print(label_names)
print(category_names)

# predicting the income column from the original unsorted array to 
# maintain original order of records

income = features_floats[:, 3]
income = np.reshape(income, (-1,1))
income_kmeans.predict(income)     
labels = income_kmeans.predict(income) 
print(labels)

# assigning the correct category name to records 

income_category = []

for label in labels:
    if str(label) == label_names[0]:
        income_category.append("low")
    elif str(label) == label_names[1]:
        income_category.append("medium")
    elif str(label) == label_names[2]:
        income_category.append("high")
print(income_category)
        
# create csv file containing Income column and Income_Category column
newCSV = []
newCSV.append(["Income", "Income_Category"])
for i in range(len(features)):
    newCSV.append([features[i, 3], income_category[i]])
np.savetxt('Income_IncomeCategory.csv', newCSV, delimiter=',', fmt='%s')
print(newCSV)

# prepare income for plotting by flattening array
income = income.flatten()

# visualizing the clusters 
plt.subplots(figsize=(10,5))

for i in range(len(labels)):
    
    if str(labels[i]) == label_names[0]:
        c1 = plt.scatter(income[i], 0, c='g', marker=".")
        
    elif str(labels[i]) == label_names[1]:
        c2 = plt.scatter(income[i], 0, c='b', marker=".")
    
    elif str(labels[i]) == label_names[2]:
        c3 = plt.scatter(income[i], 0, c='r', marker=".")
        
plt.legend([c1, c2, c3], category_names)
plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False) 
plt.xlabel("Income")
plt.ylim(0, 0)
plt.title("Income Clusters by K-Means")
plt.show()



         


        
    

