#include "../../include/synaptic.hpp"
#include <gtest/gtest.h> // Include the GoogleTest header
#include <stdexcept>
#include <memory>
#include <cmath>
#include <cstdlib>
#include <ctime>
using namespace synaptic;

class nn : public layers::module<float>
    {   
        public:
        

        layers::linear<float> input_layer = layers::linear<float>(1,16);
        layers::linear<float> hidden_layer1 = layers::linear<float>(16,8);
        layers::linear<float> hidden_layer2 = layers::linear<float>(8,4);
        layers::linear<float> output_layer = layers::linear<float>(4,1);
        connections::relu<float> act = connections::relu<float>();
        connections::tanh<float> final_act = connections::tanh<float>();

        void register_modules();
        
        std::shared_ptr<tensor<float>> forward(std::shared_ptr<tensor<float>> input);

    };

void nn::register_modules()
{
    this->optimisation_targets.insert(this->input_layer.weights);
    this->optimisation_targets.insert(this->input_layer.biases);
    this->optimisation_targets.insert(this->hidden_layer1.weights);
    this->optimisation_targets.insert(this->hidden_layer1.biases);
    this->optimisation_targets.insert(this->hidden_layer2.weights);
    this->optimisation_targets.insert(this->hidden_layer2.biases);
    this->optimisation_targets.insert(this->output_layer.weights);
    this->optimisation_targets.insert(this->output_layer.biases);
}

std::shared_ptr<tensor<float>> nn::forward(std::shared_ptr<tensor<float>> input)
{
    std::vector<std::shared_ptr<tensor<float>>> res;
    res.push_back(this->input_layer.forward(input));
    res.push_back(this->act.forward(res[res.size()-1]));
    //std::cout << *res[res.size()-1] << std::endl;
    //std::cout <<&this->hidden_layer1.weights <<std::endl;
    //std::cout<<*this->hidden_layer1.forward(res[res.size()-1])<<std::endl;
    res.push_back(this->hidden_layer1.forward(res[res.size()-1]));
    res.push_back(this->act.forward(res[res.size()-1]));
    res.push_back(this->hidden_layer2.forward(res[res.size()-1]));
    res.push_back(this->act.forward(res[res.size()-1]));
    res.push_back(this->output_layer.forward(res[res.size()-1]));
    res.push_back(this->final_act.forward(res[res.size()-1]));
    return res[res.size()-1];
}

TEST(TensorTest, NormalNeuralNetworkTest)
{   
    

    auto input_train = std::make_shared<tensor<float>>(std::vector<int>{1000,1});
    auto output_train = std::make_shared<tensor<float>>(std::vector<int>{1000,1});
    srand(42);
    for(int i=0;i<input_train->total;i++)
    {
        float rad = (float)(-10 + ((float)rand()/RAND_MAX)*(20));
        input_train->data[i] = rad;
        output_train->data[i] =std::sin(rad); 
    }


    auto input_test = std::make_shared<tensor<float>>(std::vector<int>{100,1});
    auto output_test = std::make_shared<tensor<float>>(std::vector<int>{100,1});
    srand(time(NULL));
    for(int i=0;i<input_test->total;i++)
    {
        float rad = (float)(-10 + ((float)rand()/RAND_MAX)*(20));
        input_test->data[i] = rad;
        std::cout <<"sin: "<< std::sin(rad) <<std::endl;
        output_test->data[i] = std::sin(rad); 
    }

    float total_loss=0.0;

    
    auto model = nn();
    
    auto optim = optimisers::gd<float>(model.optimisation_targets,0.1);

    auto loss_fn = loss_fn::mse<float>();
    int epochs=200;
    int iters = 4;
    int batch_size=16;
    std::vector<float> losses;
    for(int epoch=1;epoch<=epochs;epoch++)
    {
        total_loss = 0.0;
        for(int iter=1;iter<=iters;iter++)
        {
            auto x = std::make_shared<tensor<float>>(std::vector<int>{batch_size,1});
            auto y = std::make_shared<tensor<float>>(std::vector<int>{batch_size,1});
            for(int i = 0;i<x->total;i++)
            {
                int idx = rand()%1000;
                x->data[i] = input_train->data[idx];
                y->data[i] = output_train->data[idx];
            }
            
            auto res = model.forward(x);
            std::cout << "model preds: "<<*res<<std::endl;
            auto loss = loss_fn.forward(res,y);
            std::cout<<"loss: "<<loss->data[0]<<std::endl;
            loss->backprop();
            optim.step();
            optim.zero_grad();
            total_loss+=loss->data[0];
        }
        std::cout << "avg loss: "<<total_loss/iters<<std::endl;
        losses.push_back(total_loss/iters);
        
    }
    for(auto ele:losses)
    {
        std::cout << ele <<std::endl;
    }

    auto test_predictions = model.forward(input_test);
    auto test_loss = loss_fn.forward(test_predictions, output_test);
    std::cout << "Test MSE: " << test_loss->data[0] << std::endl;

    // Loop through the test predictions for detailed output
    for (int i = 0; i < input_test->total; i++) {
        std::cout << "Test Input: " << input_test->data[i] 
                  << " | Predicted: " << test_predictions->data[i] 
                  << " | Actual: " << output_test->data[i] << std::endl;
    }

    EXPECT_EQ(1,1);
}