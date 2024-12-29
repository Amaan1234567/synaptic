#include "../../include/synaptic.hpp"
#include <vector>
#include <memory>
#include <gtest/gtest.h> 
#include <stdexcept>

using namespace synaptic;

TEST(TensorTest, TensorExp)
{
    auto t1 = std::make_shared<tensor<float>>(std::vector<int>{2});
    t1->data = {1.0, 3.0};

    auto res = tensor<float>::exp(t1);
    std::vector<float> expected = {2.7183, 20.0855};

    for (int i = 0; i < res->total; i++) {
        EXPECT_NEAR(res->data[i], expected[i],0.001);

    }
}


TEST(TensorTest, TensorExpBackpropCheck)
{
    auto t1 = std::make_shared<tensor<float>>(std::vector<int>{2});
    t1->data = {-1.0, 5.0};

    auto res = tensor<float>::exp(t1);
    std::vector<float> expected = {0.3679, 148.4132};
    res->backprop();
    //std::cout<< *t1 <<std::endl;

    for (int i = 0; i < res->total; i++) {
        EXPECT_NEAR(res->data[i], expected[i],0.001);
        EXPECT_NEAR(t1->grad[i], expected[i],0.001);
    }
}