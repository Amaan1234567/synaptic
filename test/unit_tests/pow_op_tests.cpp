#include "../../include/tensor.hpp"
#include <vector>
#include <memory>
#include <gtest/gtest.h> // Include the GoogleTest header
#include <stdexcept>

TEST(TensorTest, TensorRaiseToPow)
{
    auto t1 = std::make_shared<tensor<float>>(std::vector<int>{2});
    t1->data = {1.0, 3.0};

    auto res = tensor<float>::pow(t1,2);
    std::vector<float> expected = {1.0, 9.0};

    for (int i = 0; i < res->total; i++) {
        EXPECT_FLOAT_EQ(res->data[i], expected[i]);

    }
}

TEST(TensorTest, TensorRaiseToPowFloat)
{
    auto t1 = std::make_shared<tensor<float>>(std::vector<int>{2});
    t1->data = {1.0, 3.0};

    auto res = tensor<float>::pow(t1,2.5);
    std::vector<float> expected = {1.0, 15.5885};

    for (int i = 0; i < res->total; i++) {
        EXPECT_NEAR(res->data[i], expected[i],0.001);

    }
}


TEST(TensorTest, TensorRaiseToPowNeg1WithBaseZero)
{
    auto t1 = std::make_shared<tensor<float>>(std::vector<int>{2});
    t1->data = {0.0, 2.0};

    auto res = tensor<float>::pow(t1,-1);
    std::vector<float> expected = {std::numeric_limits<float>::infinity(), 0.5};

    for (int i = 0; i < res->total; i++) {
        EXPECT_FLOAT_EQ(res->data[i], expected[i]);

    }
}


TEST(TensorTest, TensorRaiseToPowBackpropCheck)
{
    auto t1 = std::make_shared<tensor<float>>(std::vector<int>{2});
    t1->data = {1.0, 2.0};

    auto res = tensor<float>::pow(t1,2);
    std::vector<float> expected1 = {1.0,4.0};
    std::vector<float> expected2 = {2.0,4.0};
    res->backprop();
    std::cout<< *t1 <<std::endl;
    for (int i = 0; i < res->total; i++) {
        EXPECT_FLOAT_EQ(res->data[i], expected1[i]);
        EXPECT_FLOAT_EQ(t1->grad[i], expected2[i]);
    }
}