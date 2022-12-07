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
        fin.open("C:\\Users\\Virendra\\Dropbox\\My PC (LAPTOP-BOJFARUA)\\Documents\\GitHub\\MLframework\\train_dataset.csv", ios::in);

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

    friend class LinearRegression;
};

class LinearRegression : private Data
{
    float bias, bias_gradient;
    vector<float> weights, weight_gradients;

    // Function to calculate the loss
    float meanSquaredError(float y_true, float y_predicted, int total_samples)
    {
        float loss = pow((y_true - y_predicted), 2) / total_samples;
        return loss;
    }

    void initializeWeightsAndGradients()
    {
        for (int i = 0; i < features[0].size(); i++)
        {
            weights.push_back(1);
            weight_gradients.push_back(0);
        }
    }

    float computingPrediction(int i)
    {
        float y_predicted = 0;
        for (int j = 0; j < weights.size(); j++)
        {
            y_predicted += (weights[j] * features[i][j]);
        }
        y_predicted += this->bias;
        return y_predicted;
    }

    void updatingGradients(float y_predicted, int row)
    {
        this->bias_gradient = 0;
        weight_gradients.clear();

        this->bias_gradient += ((-2 * (target[row] - y_predicted))/target.size());

        for (int col = 0; col < features[0].size(); col++)
        {
            weight_gradients[col] += ((-2 * ((target[row] - y_predicted) * features[row][col]))/target.size());
        }
    }

    void updatingWeightsAndBias(float learning_rate)
    {
        for (int j = 0; j < weights.size(); j++)
        {
            weights[j] = weights[j] - ((learning_rate) * (weight_gradients[j]));
        }
        this->bias = this->bias - ((learning_rate) * (bias_gradient));
    }

    void gradientDescent(float learning_rate, int epochs)
    {
        int total_samples = target.size();
        float current_loss = __FLT_MAX__;

        initializeWeightsAndGradients();

        // Run for the number of epochs specified to train the model
        for (int e = 0; e < epochs; e++)
        {
            for (int i = 0; i < total_samples; i++)
            {
                // Initializing the weights and bias
                float y_predicted = computingPrediction(i);
                // Calculate the loss
                float loss = meanSquaredError(target[i], y_predicted, total_samples);

                // cout << "Delta: " << target[i]-y_predicted << endl;

                // Compare the present loss with the previous
                // If less than the previous then try to further optimize
                // Else skip optimization
                if (loss < current_loss)
                {
                    updatingGradients(y_predicted, i);
                    updatingWeightsAndBias(learning_rate);
                    current_loss = loss;
                }

                // Here print the weights and bias and loss for every iteration
            }
            cout << "Epoch: " << e << "\t"
                 << "Loss: " << current_loss << endl;
        }
    }

public:
    LinearRegression()
    {
        this->bias = 0;
        readCSV();
    }
    // Enter the learning rate based on how fast you want your model to learn
    void fit(float learning_rate, int epochs)
    {
        this->gradientDescent(learning_rate, epochs);
    }

    // Returns the predicted value on providing the necessary info
    float predict(int size, vector<float> features)
    {
        float y_predicted = 0;
        for (int i = 0; i < weights.size(); i++)
        {
            y_predicted += (weights[i] * features[i]);
        }
        y_predicted += this->bias;
        return y_predicted;
    }

    void score()
    {
        float ctr = 0;
        for (int i = 0; i < target.size(); i++)
        { 
            float predicted_value = predict(features[0].size(), features[i]);
            float difference = target[i]*0.75;
            float ul = target[i] + difference;
            float ll = target[i] - difference;
            if ((predicted_value > ll) && (predicted_value < ul))
            {
                ctr++;
            }
        }
        float score = ctr / target.size();
        cout << "Score: " << score << endl
             << ctr << endl;
    }

    void showWeightsAndBias()
    {
        for (int i = 0; i < weights.size(); i++)
        {
            cout << "weight " << i + 1 << ": " << weights[i] << endl;
        }
        cout << "bias: " << this->bias;
    }
};

class GUI
{
    // GUI Code here
    
};

int main()
{
    LinearRegression nn;
    nn.fit(0.0001, 100);
    nn.score();
    nn.showWeightsAndBias();
    return 0;
}