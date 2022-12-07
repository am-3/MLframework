#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <imgui.h>
using namespace std;

class Storage
{
    vector<string> heading;

    // Dataset for training
    vector<vector<float>> dataset;
    vector<float> result;

    // Dataset for testing
    vector<vector<float>> testing;
    vector<float> target;

    // Importing and exporting func
    void import_csv(string path, int flag)
    {
        vector<vector<float>> features;
        vector<float> answers;

        // File pointer
        fstream fin;

        // Open an existing file
        fin.open(path, ios::in);

        vector<float> temp;
        string word, line;
        int heading = 0;
        while (fin >> line)
        {

            temp.clear();

            stringstream s(line);

            while (getline(s, word, ','))
            {
                float val = stof(word);
                temp.push_back(val);
            }

            dataset.push_back(temp);
        }

        for (int i = 0; i < dataset.size(); i++)
        {
            answers.push_back(dataset[i].back());
            dataset[i].pop_back();
        }

        if (flag == 0)
        { // Training Dataset

            this->dataset = features;
            this->result = answers;
        }
        else
        {
            // Testing Dataset
            this->testing = features;
            this->target = answers;
        }
    }

public:
    Storage(string trainset, string testset)
    {
        import_csv(trainset, 0);

        //import_csv(testset, 1);
    }
};

class GUI
{
    // GUI Code here
};

// Driver Code
int main()
{
    cout << "Running..." << endl;

    //Storage("train.csv", "test.csv");

    GUI();
    return 0;
}
