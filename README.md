# Project: Machine Learning Framework

## Team: Atharva More & Virendra Panchal

Project for OOPs Lab



### Problem Statement

Building a framework for Machine Learning as a library. 

Demonstrating using House Price Prediction example.





### UML

![uml](\imgs\uml.jpeg)





### Assumptions

Input data is given in csv format.

Input data is numeric in nature.

Input data has last column/field as its result/target.





### Methodology

##### Inputs: 

CSV file with n columns; of which n-1 are features and the nth column is target.



##### Training the model:

To train the model, Stochastic Gradient Descent algorithm is applied. In this algorithm, one datapoint is taken at a time for training the model. Appropriate weights are multiplied with the features and added up along with bias. 

![gdgif](\imgs\gradient_descent_gif.gif)

This gives the predicted value. The loss between true and predicted values is calculated. Further to reduce the loss, first the gradient of the loss is obtained *(which is the derivate of the loss function with respect to the weights and bias)* and then the obtained gradients are multiplied with the learning rate; and subtracted from the corresponding weights and bias.



##### Result:

A Linear Regression model is created that predicts the house price with user given features.



#### Terminologies and techniques used:

![lr](\imgs\linear_regression_graph2.png)

**Linear Regression:**  Linear Regression is a linear approach for modelling the relationship between a scalar response and one or more explanatory variables (also known as dependent and independent variables). Basically, we plot all the datapoints on a graph and initially a random line (y = mx + c) is drawn which may or may not pass through all the points. The goal is to make the line pass through majority of the datapoints. For that loss is calculated which is the sum of the difference of the actual value and the predicted value (y) of all datapoints. This loss is reduced by changing the parameters of the line (m and c). Finally, a value of m and c is reached such that the loss value doesn't decrease further. This value gives the best line possible.
Here, m is called weight as every feature does have some weightage to the predicted value and c is called bias.

**Learning Rate: ** How fast/slow the model should correct itself to minimise the loss between actual and predicted values.

**Gradient Overshooting:-** While training the gradient may or may not overshoot i.e. take a much larger value than expected. So to avoid that we use gradient clipping to make the gradient remain within a range of values.



#### Testing the model: 

For testing the model, the user enters the feature values accordingly. These are then multiplied with their corresponding weights and added up along with the bias. This gives the predicted result value.





### Usage

Ensure that the g++ compiler and git is set up in your machine.

You can check if it is ready to go by running ,

```shell
> g++
```

and seeing a similar output on the console.

![g++](\imgs\usuage_g++.png)



Now open a terminal/command prompt instance where you want to store the code.

```shell
> cd `target directory`
```

**Note**: Ensure that this directory is empty.



Now let's grab the code from GitHub,

```shell
> git clone https://github.com/am-3/MLframework.git .
```

The code along with the sample datasets will be downloaded.



Let's compile the binaries,

```shell
> g++ Stochastic_Gradient_Descent.cpp -o SGD.exe `output name inplace of SGD`
```



Finally, to execute

On Linux,

```shell
> ./SGD.exe
```



On Windows,

```shell
> .\SGD.exe
```





### Sample Output

![output](\imgs\output.png)





### Sample Dataset

| Area | Location | Bedrooms | Price    |
| ---- | -------- | -------- | -------- |
| 400  | 1        | 1        | 6200000  |
| 1000 | 1        | 2        | 9500000  |
| 1245 | 1        | 2        | 14900000 |
| 1183 | 1        | 2        | 14000000 |
| 1245 | 1        | 2        | 3600000  |
| 495  | 1        | 1        | 6400000  |
| 495  | 1        | 1        | 3800000  |
| 1050 | 1        | 2        | 15500000 |
| 600  | 1        | 1        | 7000000  |
| 600  | 1        | 1        | 6694000  |
| .... 7190 more rows          |
