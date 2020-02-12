"""
@author: Kaustubh
Wine Quality Prediction
"""
#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

print("-------------------------------------------------------------------")
print("------------------------ Implementing KNN Model -------------------")
print("-------------------------------------------------------------------")
#Importing the dataset
dataset = pd.read_csv('wine_data.csv')
X = dataset.iloc[: , :-1]
Y = dataset.iloc[: , 11]

#Checking the null or missing value
from sklearn.impute import SimpleImputer
missingvalues = SimpleImputer(missing_values = np.nan , strategy = 'mean' , verbose = 0)
missingvalues = missingvalues.fit(X)
X = missingvalues.transform(X)

#Apply test train split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = .2, random_state = 0)

#Apply Feature Scalling: As Machine learning model is based on euclidian distances therefore it is 
# a required step to follow before applying any model
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.fit_transform(X_test)

#Applied feature extraction using PCA: to check the variance
from sklearn.decomposition import PCA
"""
pca = PCA(n_components = None)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
"""
#First used above code to find the variance and found that we can reduce total independent variable to 7
pca = PCA(n_components = 7)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

#Knn Model: Used metric = 'minkowski' , p = 2 for defining eucledian distance method
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors =1, metric = 'minkowski' , p = 2)
classifier.fit(X_train, y_train)

#Predicting the result
y_pred = classifier.predict(X_test)

#Generating Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test, y_pred)

#Calculating the accuracy
correct = 0
total = y_pred.size
for i in range(0, confusion.size):
    for j in range(confusion[0].size):
        if i == j:
            correct = correct + confusion[i][j]

Accuracy = correct / total
print("Predicted Accuracy: " ,Accuracy , "\n")

print("Applying K fold cross validation")
#Validating the accuracy using k fold cross validation
from sklearn.model_selection import cross_val_score
accuracy = cross_val_score(estimator = classifier , X = X_train , y = y_train, cv = 10)
print("Mean accuracy: " , accuracy.mean())
print("Std deviation: " , accuracy.std())

"""
Final Comment:
The accuracy obtained by our model is 52 % which is in the range of mean +- std 
To make the best selection of parameter in the model we will work on grid search
"""
#Applying grid selection
from sklearn.model_selection import GridSearchCV
parameters = [{'n_neighbors' :[1,3,5,7,9,12,15,20] , 'metric' :['minkowski'] , 'p' :[2] }
              ]
grid_search_knn = GridSearchCV(estimator = classifier,
                    param_grid = parameters,
                    scoring = 'accuracy',
                    cv = 10,
                    n_jobs = -1)

grid_search_knn = grid_search_knn.fit(X_train, y_train)
best_accuracy_knn = grid_search_knn.best_score_
best_parameter_knn = grid_search_knn.best_params_
print("Optimal Parameter -- knn model: " , "\n", best_parameter_knn)
"""
Changed n_neighbors in the knn model from 5 to 1 as it gives the best accuracy
"""


print("--------------------------------------------------------------------")
print("---------------- Implementing Random Forest Model ------------------")
print("--------------------------------------------------------------------")

#Applying the model where n_estimator is number of trees in the model
"""
Used n_estimator = 10 and criterion = 'entropy' but by optimizing paramters of random forest
classifier using grid search changed it to 50 and gini
"""
from sklearn.ensemble import RandomForestClassifier
randomForestClassifier = RandomForestClassifier(n_estimators = 50, criterion = 'gini' , random_state = 0)
randomForestClassifier.fit(X_train, y_train)

#Predicting the value of test data
random_y_pred = randomForestClassifier.predict(X_test)

#Implementing the confusion metrics
random_cm = confusion_matrix(y_test, random_y_pred)

#Calculating the Accuracy
correct = 0
total = y_pred.size
for i in range(0, random_cm.size):
    for j in range(random_cm[0].size):
        if i == j:
            correct = correct + random_cm[i][j]

Accuracy = correct / total
print("Predicted Accuracy: " ,Accuracy , "\n")

print("Applying K fold cross validation")
#Validating the accuracy using k fold cross validation
randomForrestAccuracy = cross_val_score(estimator = randomForestClassifier,X = X_train,y = y_train, cv= 10)
print("Mean Accuracy: " ,randomForrestAccuracy.mean() )
print("Std deviation: " ,randomForrestAccuracy.std() )

"""
Accuracy lies below the model predicted accuracy hence we need to adjust the parameters.
using grid search we will adjust the parameter.
"""

random_parameters = [{'n_estimators' : [48 ,49, 50 , 51] , 'criterion' : ['entropy', 'gini']}
                    ]

grid_search_random = GridSearchCV(estimator = randomForestClassifier,
                    param_grid = random_parameters,
                    scoring = 'accuracy',
                    cv = 10,
                    n_jobs = -1)

grid_search_random = grid_search_random.fit(X_train, y_train)
best_accuracy_random = grid_search_random.best_score_
best_parameter_random = grid_search_random.best_params_
print("Optimal Parameter -- random forest model: " , "\n" , best_parameter_random)

print("--------------------------------------------------------------------")
print("------------------ Implementing Naive Bayes Model ------------------")
print("--------------------------------------------------------------------")

#Implementing the model
"""
We didn't used any parameter to train the model but using grid search we found that 
var_smoothing is not the default value that is 1e-09 but it should be 1e-08
"""
from sklearn.naive_bayes import GaussianNB
naiveClassifier = GaussianNB(var_smoothing = 1e-08)
naiveClassifier.fit(X_train, y_train)

#Predicting the result
y_pred_NB = naiveClassifier.predict(X_test)

#Building confusion matrix
naive_cm = confusion_matrix(y_test, y_pred_NB)

#Calculating the Accuracy
correct = 0
total = y_pred.size
for i in range(0, naive_cm.size):
    for j in range(naive_cm[0].size):
        if i == j:
            correct = correct + naive_cm[i][j]

Accuracy = correct / total
print("Predicted Accuracy: " ,Accuracy , "\n")

print("Applying K fold cross validation")
#Validating the accuracy using k fold cross validation
naiveAccuracy = cross_val_score(estimator = naiveClassifier,X = X_train,y = y_train, cv= 10)
print("Mean Accuracy: " ,naiveAccuracy.mean() )
print("Std deviation: " ,naiveAccuracy.std() )

"""
Accuracy lies below the model predicted accuracy hence we need to adjust the parameters.
using grid search we will adjust the parameter.
"""

naive_parameters = [{'var_smoothing' : [1e-08 , 1e-09 , 1e-10]}
                    ]

grid_search_naive = GridSearchCV(estimator = naiveClassifier,
                    param_grid = naive_parameters,
                    scoring = 'accuracy',
                    cv = 10,
                    n_jobs = -1)

grid_search_naive = grid_search_naive.fit(X_train, y_train)
best_accuracy_naive = grid_search_naive.best_score_
best_parameter_naive = grid_search_naive.best_params_
print("Optimal Parameter -- random forest model: " , "\n" , best_parameter_naive)


"""
We comapared the result and based on the predicted accuracy and average mean of the 
k fold cross validation we can say that Random Forest Classification was the best ML
algorithm among all the applied one in this project. When training, each tree in a 
random forest learns from a random sample of the data points. The samples are drawn 
with replacement, known as bootstrapping which allows random forest to predict with 
the very good accuracy (As it predict on the majority vote from the n number trees 
generated).
"""
