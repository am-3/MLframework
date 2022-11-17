#include <iostream>
#include <bits/stdc++.h>
using namespace std;

float sigmoid(float x){
    return 1/(1+exp(x));
}

float log_loss(float y_true[], float y_predicted[]){
    float epsilon = 1e-15;

    int len_y_true = sizeof(y_true)/sizeof(y_true[0]);
    int len_y_predicted = sizeof(y_predicted)/sizeof(y_predicted[0]);

    float *y_predicted_new = new float[len_y_predicted];

    for(int i = 0; i < len_y_predicted; i++){
        if(y_predicted[i] > epsilon){
            y_predicted_new[i] = y_predicted;
        }
        else{
            y_predicted_new[i] = epsilon;
        }
    }

    for(int i = 0; i < len_y_predicted; i++){
        if(y_predicted_new[i] < (1-epsilon)){
            y_predicted_new[i];
        }
        else{
            y_predicted_new[i] = (1-epsilon);
        }
    }


}