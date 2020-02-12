# Wine-Quality-Prediction
Based on the physio-chemical tests the data set has 11 independent variable and 1 output variable for a wine where the output variable is quality of the wine with score 0 is worst and score of 10 is the best. Required to build at-least 2 and predict the results. 

These datasets can be viewed as classification or regression tasks. The classes are ordered and not balanced (e.g. there are munch more normal wines than excellent or poor ones). Also, as we are not sure if all input variables are relevant So, we applied PCA to extract the relevant feature out of the 11 independent variable.

No of instance: 4898

 Input variables (based on physicochemical tests):
 
   1 - fixed acidity
   2 - volatile acidity
   3 - citric acid
   4 - residual sugar
   5 - chlorides
   6 - free sulfur dioxide
   7 - total sulfur dioxide
   8 - density
   9 - pH
   10 - sulphates
   11 - alcohol
   
Output variable (based on sensory data): 

   12 - quality (score between 0 and 10)

Final Result:
We comapared the result and based on the predicted accuracy and average mean of the k fold cross validation we can say that Random Forest Classification was the best ML algorithm among all the applied one in this project. When training, each tree in a random forest learns from a random sample of the data points. The samples are drawn with replacement, known as bootstrapping which allows random forest to predict with the very good accuracy (As it predict on the majority vote from the n number of trees generated).

Instruction to Run:
Paste the dataset and code in the same directory and change the address of dataset in the code as per your location
Compile the python file: Jupyter or terminal
