#include "../../include/synaptic.hpp"
#include <gtest/gtest.h> // Include the GoogleTest header
#include <stdexcept>
#include <memory>
#include <cmath>
#include <cstdlib>
#include <ctime>

#ifndef M_PIf32
double M_PIf32 = 3.141592653589793238462643383279502884;
#endif

using namespace synaptic;

class nn : public layers::module<float>
    {   
        public:
        

        layers::linear<float> input_layer = layers::linear<float>(1,32);
        layers::linear<float> hidden_layer1 = layers::linear<float>(32,16);
        layers::linear<float> output_layer = layers::linear<float>(16,1);
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
        float rad = (float)(-M_PIf32 + ((float)rand()/RAND_MAX)*(2*(M_PIf32)));
        input_train->data[i] = rad;
        output_train->data[i] =std::sin(rad); 
    }


    auto input_test = std::make_shared<tensor<float>>(std::vector<int>{100,1});
    auto output_test = std::make_shared<tensor<float>>(std::vector<int>{100,1});
    for(int i=0;i<input_test->total;i++)
    {
        float rad = (float)(-M_PIf32 + ((float)rand()/RAND_MAX)*(2*(M_PIf32)));
        input_test->data[i] = rad;
        std::cout <<"sin: "<< std::sin(rad) <<std::endl;
        output_test->data[i] = std::sin(rad); 
    }

    float total_loss=0.0;

    
    auto model = nn();
    
    auto optim = optimisers::gd<float>(0.005);
    model.register_modules();
    auto loss_fn = loss_fn::mse<float>();
    int epochs=2000;
    int iters = 1;
    int batch_size=32;
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
            optim.step(model.optimisation_targets);
            optim.zero_grad(model.optimisation_targets);
            total_loss+=loss->data[0];
        }
        auto test_predictions = model.forward(input_test);
        auto test_loss = loss_fn.forward(test_predictions, output_test);
        std::cout << "Test MSE: " << test_loss->data[0] << std::endl;
        //std::cout << "avg loss: "<<total_loss/iters<<std::endl;
        losses.push_back(total_loss/iters);
        
    }
    for(auto ele:losses)
    {
        std::cout << ele <<std::endl;
    }
    auto test_predictions = model.forward(input_test);
    auto test_loss = loss_fn.forward(test_predictions, output_test);
    std::cout <<"train loss: "<<std::accumulate(losses.begin(),losses.end(),0.0)/losses.size()<<std::endl;
    std::cout << "Test MSE: " << test_loss->data[0] << std::endl;

    // Loop through the test predictions for detailed output
    for (int i = 0; i < input_test->total; i++) {
        std::cout << "Test Input: " << input_test->data[i] 
                  << " | Predicted: " << test_predictions->data[i] 
                  << " | Actual: " << output_test->data[i] << std::endl;
    }

    EXPECT_EQ(1,1);
}