#include "../../include/synaptic.hpp"
#include <gtest/gtest.h>

TEST(TensorTest, TensorRelu)
{
    auto t4 = std::make_shared<synaptic::tensor<float>>(std::vector<int>{5, 3, 2});
    t4->data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30};

    synaptic::connections::relu<float> relu = synaptic::connections::relu<float>();
    auto res = relu.forward(t4);
    //std::cout << *res << std::endl;
    std::vector<float> expected = std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30};

    for (int i = 0; i < res->dims.size(); i++)
    {
        EXPECT_EQ(res->data[i], expected[i]);
    }
}


TEST(TensorTest, TensorReluWithParams)
{
    auto t4 = std::make_shared<synaptic::tensor<float>>(std::vector<int>{5, 2});
    t4->data = {0.2729,  0.9562, -1.0154, -0.7500, -0.0861,  0.2006,  1.2051, -1.0691,
         1.7741,  0.8285};

    synaptic::connections::relu<float> relu = synaptic::connections::relu<float>(2,-1);
    auto res = relu.forward(t4);
    //std::cout << *res << std::endl;
    std::vector<float> expected = std::vector<float>{0.5457,  1.9124, -1.0000, -1.0000, -1.0000,  0.4011,  2.4102, -1.0000,
         3.5482,  1.6569};

    for (int i = 0; i < res->dims.size(); i++)
    {
        EXPECT_NEAR(res->data[i], expected[i],0.001);
    }
}


TEST(TensorTest, TensorReluBackpropCheck)
{
    auto t4 = std::make_shared<synaptic::tensor<float>>(std::vector<int>{2, 2});
    t4->data = {1, -2, 3, 4};

    synaptic::connections::relu<float> relu = synaptic::connections::relu<float>();
    auto res = relu.forward(t4);
    //std::cout << *res << std::endl;
    std::vector<float> expected = std::vector<float>{1.0,0.0,1.0,1.0};

    res->backprop();
    for (int i = 0; i < res->total; i++)
    {
        EXPECT_EQ(t4->grad[i], expected[i]);
    }
}


TEST(TensorTest, TensorReluBackpropCheckWithParams)
{
    auto t4 = std::make_shared<synaptic::tensor<float>>(std::vector<int>{5, 2});
    t4->data = {0.2729,  0.9562, -1.0154, -0.7500, -0.0861,  0.2006,  1.2051, -1.0691,
         1.7741,  0.8285};

    synaptic::connections::relu<float> relu = synaptic::connections::relu<float>(2,-1);
    auto res = relu.forward(t4);
    //std::cout << *res << std::endl;
    std::vector<float> expected = std::vector<float>{2., 2., -1., -1., -1., 2., 2., -1., 2., 2.};

    res->backprop();
    for (int i = 0; i < res->total; i++)
    {
        EXPECT_NEAR(t4->grad[i], expected[i],0.001);
    }
}