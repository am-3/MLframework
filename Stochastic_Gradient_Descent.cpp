#include <iostream>
#include <math.h>
#include <vector>
#include <limits>
#include <fstream>
#include <sstream>
#include <imgui.h>
using namespace std;

class Data
{
    vector<vector<float>> features;
    vector<float> target;
    void readCSV()
    {
        fstream fin;
        fin.open("E:\\VS Code Programs\\OOP_Assignments\\train.csv", ios::in);

        vector<float> temp_vec;
        string word, line;
        int ctr = 0;

        while (fin >> line)
        {
            if (ctr == 0)
            {
                ctr++;
                continue;
            }

            temp_vec.clear();

            stringstream s(line);

            while (getline(s, word, ','))
            {
                float val = stof(word);
                temp_vec.push_back(val);
            }
            features.push_back(temp_vec);
        }

        for (int i = 0; i < features.size(); i++)
        {
            target.push_back(features[i].back());
            features[i].pop_back();
        }
    }

    friend class NeuralNetwork;
};

class NeuralNetwork : private Data
{
    float w1, w2, w3, w4, w5, bias;

    // Function to calculate the loss
    float meanSquaredError(float y_true, float y_predicted, int total_samples)
    {
        float loss = pow((y_true - y_predicted), 2) / total_samples;
        return loss;
    }

    void gradientDescent(vector<vector<float>> X_train, vector<float> y_train, float learn_rate)
    {
        float learning_rate = learn_rate;
        int total_samples = y_train.size();
        float current_loss = __FLT_MAX__;

        // Run for the number of epochs specified to train the model
        for (int i = 0; i < total_samples; i++)
        {
            float y_predicted = this->w1 * X_train[i][0] + this->w2 * X_train[i][1] + this->w3 * X_train[i][2] + this->w4 * X_train[i][3] + this->w5 * X_train[i][4] + this->bias;

            float loss = meanSquaredError(y_train[i], y_predicted, total_samples);

            if (loss < current_loss)
            {
                // Get the differentitated values of the weights and the bias
                float w1_diff = -(2 / total_samples) * ((y_train[i] - y_predicted) * X_train[i][0]);
                float w2_diff = -(2 / total_samples) * ((y_train[i] - y_predicted) * X_train[i][1]);
                float w3_diff = -(2 / total_samples) * ((y_train[i] - y_predicted) * X_train[i][2]);
                float w4_diff = -(2 / total_samples) * ((y_train[i] - y_predicted) * X_train[i][3]);
                float w5_diff = -(2 / total_samples) * ((y_train[i] - y_predicted) * X_train[i][4]);
                float bias_diff = -(2 / total_samples) * (y_train[i] - y_predicted);

                // Change the weights and the bias to improve the model
                this->w1 = this->w1 - (learning_rate)*w1_diff;
                this->w2 = this->w2 - (learning_rate)*w2_diff;
                this->w3 = this->w3 - (learning_rate)*w3_diff;
                this->w4 = this->w4 - (learning_rate)*w4_diff;
                this->w5 = this->w5 - (learning_rate)*w5_diff;
                this->bias = this->bias - (learning_rate)*bias_diff;

                current_loss = loss;
            }

            // Here print the weights and bias and loss for every iteration
            cout << "Sample No: " << i << "\t"
                 << "Loss: " << current_loss << endl;
        }
    }

public:
    NeuralNetwork()
    {
        this->w1 = 1;
        this->w2 = 1;
        this->w3 = 1;
        this->w4 = 1;
        this->w5 = 1;
        this->bias = 0;
        readCSV();
    }
    // Enter the learning rate based on how fast you want your model to learn
    void fit(float learning_rate)
    {
        this->gradientDescent(features, target, learning_rate);
    }

    // Returns the predicted value on providing the necessary info
    float predict(float neighbourhood, float floor_space, float num_of_bhks, float num_of_parking_slots, float balcony_area)
    {
        float y_predicted = this->w1 * neighbourhood + this->w2 * floor_space + this->w3 * num_of_bhks + this->w4 * num_of_parking_slots + this->w5 * balcony_area + this->bias;
        return y_predicted;
    }

    void score(){
        int ctr = 0;
        for(int i = 0; i < target.size(); i++){
            float predicted_value = predict(features[i][0], features[i][1], features[i][2], features[i][3], features[i][4]);
            float temp = target[i]*0.1;
            if((predicted_value > target[i] - temp) && (predicted_value < target[i] + temp)){
                ctr++;
            }
        }

        cout<<"Score: "<<ctr/target.size()<<endl;
    }
};

class GUI
{
    // GUI Code here
    
};

int main()
{
    NeuralNetwork nn;
    nn.fit(0.5);
    nn.score();
    return 0;
}