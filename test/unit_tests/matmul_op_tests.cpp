#include "../../include/synaptic.hpp"

#include <vector>
#include <memory>
#include <gtest/gtest.h>
#include <stdexcept>

using namespace synaptic;

TEST(TensorTest, MatmulOfTwoTensors)
{
    auto t1 = std::make_shared<tensor<float>>(std::vector<int>{2, 5});
    auto t2 = std::make_shared<tensor<float>>(std::vector<int>{5, 3});
    t1->data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    t2->data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
                12.0, 13.0, 14.0, 15.0};

    auto res = tensor<float>::matmul(t1, t2);
    std::vector<float> expected = {
        135., 150., 165.,
        310., 350., 390.};

    for (int i = 0; i < res->total; i++)
    {
        EXPECT_FLOAT_EQ(res->data[i], expected[i]);
    }
}

TEST(TensorTest, MatmulOfTwoTensorsBackpropCheck)
{
    auto t1 = std::make_shared<tensor<float>>(std::vector<int>{2, 5});
    auto t2 = std::make_shared<tensor<float>>(std::vector<int>{5, 3});
    t1->data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    t2->data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
                12.0, 13.0, 14.0, 15.0};

    auto res = tensor<float>::matmul(t1, t2);
    res->backprop();
    std::vector<float> expected1 = {
        6., 15., 24., 33., 42.,
         6., 15., 24., 33., 42.};
    std::vector<float> expected2={
         7.,  7.,  7.,
         9.,  9.,  9.,
        11., 11., 11.,
        13., 13., 13.,
        15., 15., 15.
    };
    for (int i = 0; i < t1->total; i++)
    {
        EXPECT_FLOAT_EQ(t1->grad[i], expected1[i]);
    }

    for (int i = 0; i < t2->total; i++)
    {
        EXPECT_FLOAT_EQ(t2->grad[i], expected2[i]);
    }
}


// Test cases where the shape mismatch triggers an assertion failure
TEST(TensorAssertionFailureTest, TensorMatmulShapeMismatch)
{
    auto t1 = std::make_shared<tensor<int>>(std::vector<int>{2, 4});
    auto t2 = std::make_shared<tensor<int>>(std::vector<int>{3, 2});
    t1->data = {1, 2, 3, 4, 5, 6 ,7 ,8};
    t2->data = {1, 2, 3, 4, 5, 6};

    EXPECT_THROW({ auto res = tensor<int>::matmul(t1,t2); }, std::runtime_error);
}

TEST(TensorAssertionFailureTest, TensorMatmulWithScalarShapeMismatch)
{
    auto t1 = std::make_shared<tensor<float>>(std::vector<int>{3});
    auto t2 = std::make_shared<tensor<float>>(std::vector<int>{1, 3});
    t1->data = {1.0f, 2.0f, 3.0f};
    t2->data = {1.0f, 2.0f, 3.0f};

    EXPECT_THROW({ auto res = t1 * t2; }, std::runtime_error);
}