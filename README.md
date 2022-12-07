Project: Machine Learning Framework

Contributors: Atharva More & Virendra Panchal

Project for OOPs Lab

Methodoly:
Inputs: CSV file with n columns; of which n-1 are features and the nth column is target.
We use Linear Regression to create a model that predicts the house price of user given features.
Linear Regression:  Linear Regression is a linear approach for modelling the relationship between a scalar response and one or more explanatory variables (also known as dependent and independent variables). Basically, we plot all the datapoints on a graph and initially a random line (y = mx + c) is drawn which may or may not pass through all the points. The goal is to make it pass through majority of the datapoints. For that first loss is calculated which is the sum of the difference of the actual value and the predicted value of all datapoints. This loss is reduced by changing the parameters of the line i.e. m and c. Finally, a value of m and c is reached such that the loss value doesn't decrease. This gives the best fit line which is then used for prediction.
Here, m is called weight as every feature does have some weightage to the predicted value and c is called bias.
Training the model:- To train the model Stochastic Gradient Descent algorithm is applied. In this one datapoint is taken at a time for training.
