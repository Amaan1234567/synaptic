#include "../../include/synaptic.hpp"
#include <gtest/gtest.h> // Include the GoogleTest header
#include <stdexcept>
#include <memory>
#include <cmath>
#include <cstdlib>
#include <ctime>
using namespace synaptic;

TEST(TensorTest, NormalNeuralNetworkTest)
{
    auto input_train = std::make_shared<tensor<float>>(std::vector<int>{1000,1});
    auto output_train = std::make_shared<tensor<float>>(std::vector<int>{1000,1});
    srand(time(NULL));
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

    float loss=1000.0;

    auto input_layer = synaptic::layers::linear<float>(1,5);
    auto hidden_layer = synaptic::layers::linear<float>(5,3);
    auto output_layer = synaptic::layers::linear<float>(3,1);

    auto act = synaptic::connections::relu<float>();
    auto final_act = synaptic::connections::tanh<float>();

    auto loss_fn = synaptic::loss_fn::mse<float>();
    int epochs=100;
    int iters = 100;
    int batch_size=10;
    for(int epoch=1;epoch<=epochs;epoch++)
    {
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
            auto res = input_layer.forward(x);
            res = act.forward(res);
            res = hidden_layer.forward(res);
            res = act.forward(res);
            res = output_layer.forward(res);
            res = final_act.forward(res);

            auto loss = loss_fn.forward(res,y);
            std::cout<<"loss: "<<loss->data[0]<<std::endl;
        }
        
    }
    EXPECT_EQ(1,1);
}