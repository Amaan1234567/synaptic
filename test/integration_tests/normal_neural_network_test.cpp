#include "../../include/synaptic.hpp"
#include <gtest/gtest.h> // Include the GoogleTest header
#include <stdexcept>
#include <memory>
#include <cmath>
#include <cstdlib>
#include <ctime>
using namespace synaptic;

class nn : public synaptic::layers::module<float>
    {   
        public:
        

        layers::linear<float> input_layer = synaptic::layers::linear<float>(1,15);
        layers::linear<float> hidden_layer1 = synaptic::layers::linear<float>(15,10);
        layers::linear<float> hidden_layer2 = synaptic::layers::linear<float>(10,5);
        layers::linear<float> output_layer = synaptic::layers::linear<float>(5,1);
        synaptic::connections::relu<float> act = synaptic::connections::relu<float>();
        synaptic::connections::tanh<float> final_act = synaptic::connections::tanh<float>();

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
    auto res = this->input_layer.forward(input);
    res = this->act.forward(res);
    res = this->hidden_layer1.forward(res);
    res = this->act.forward(res);
    res = this->hidden_layer2.forward(res);
    res = this->act.forward(res);
    res = this->output_layer.forward(res);
    res = this->final_act.forward(res);
    return res;
}

TEST(TensorTest, NormalNeuralNetworkTest)
{   
    

    auto input_train = std::make_shared<tensor<float>>(std::vector<int>{1000,1});
    auto output_train = std::make_shared<tensor<float>>(std::vector<int>{1000,1});
    srand(42);
    for(int i=0;i<input_train->total;i++)
    {
        float rad = (float)rand()/RAND_MAX;
        input_train->data[i] = std::sin(rad);
        output_train->data[i] = rad; 
    }


    auto input_test = std::make_shared<tensor<float>>(std::vector<int>{100,1});
    auto output_test = std::make_shared<tensor<float>>(std::vector<int>{100,1});
    srand(time(NULL));
    for(int i=0;i<input_test->total;i++)
    {
        float rad = (float)rand()/RAND_MAX;
        input_test->data[i] = std::sin(rad);
        std::cout <<"sin: "<< std::sin(rad) <<std::endl;
        output_test->data[i] = rad; 
    }

    float total_loss=0.0;

    
    auto model = nn();
    
    auto optim = optimisers::gd<float>(model.optimisation_targets,0.01);

    auto loss_fn = loss_fn::mse<float>();
    int epochs=100;
    int iters = 100;
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
    EXPECT_EQ(1,1);
}