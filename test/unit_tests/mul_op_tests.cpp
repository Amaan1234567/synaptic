#include "../../include/synaptic.hpp"
#include <vector>
#include <memory>
#include <gtest/gtest.h>
#include <stdexcept>

using namespace synaptic;

TEST(TensorTest, MultiplicationOfTwoTensors)
{
    auto t1 = std::make_shared<tensor<float>>(std::vector<int>{2});
    auto t2 = std::make_shared<tensor<float>>(std::vector<int>{2});
    t1->data = {1.0, 3.0};
    t2->data = {1.0, 4.0};

    auto res = t1 * t2;
    std::vector<float> expected = {1.0, 12.0};

    for (int i = 0; i < res->total; i++)
    {
        EXPECT_FLOAT_EQ(res->data[i], expected[i]);
    }
}

TEST(TensorTest, MultiplicationOfTwoTensorsBackpropCheck)
{
    auto t1 = std::make_shared<tensor<float>>(std::vector<int>{3});
    auto t2 = std::make_shared<tensor<float>>(std::vector<int>{3});

    t1->data = {1, 2, 3};
    t2->data = {4, 5, 6};
    auto c = t1 * t2;
    c->backprop();
    std::vector<float> expected1 = {4, 5, 6};
    std::vector<float> expected2 = {1, 2, 3};

    for (int i = 0; i < t1->total; i++)
    {
        EXPECT_FLOAT_EQ(t1->grad[i], expected1[i]);
        EXPECT_FLOAT_EQ(t2->grad[i], expected2[i]);
    }
}

TEST(TensorTest, MultiplicationWithScalar)
{
    auto t3 = std::make_shared<tensor<int>>(std::vector<int>{5, 3, 2});
    t3->data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30};

    auto res = t3 * 1;
    std::vector<int> expected = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30};

    for (int i = 0; i < res->total; i++)
    {
        EXPECT_EQ(res->data[i], expected[i]);
    }
}

TEST(TensorTest, MultiplicationWithScalarInt)
{
    auto t3 = std::make_shared<tensor<int>>(std::vector<int>{5, 3, 2});
    t3->data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30};

    auto res = 1 * t3;
    std::vector<int> expected = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30};

    for (int i = 0; i < res->total; i++)
    {
        EXPECT_EQ(res->data[i], expected[i]);
    }
}

// Test case 3
TEST(TensorTest, ScalarFloatMultiplicationToTensor)
{
    auto t4 = std::make_shared<tensor<float>>(std::vector<int>{5, 3, 2});
    t4->data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30};

    auto res = 1.0f * t4;
    std::vector<float> expected = {1.0000, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30};

    for (int i = 0; i < res->total; i++)
    {
        EXPECT_NEAR(res->data[i], expected[i], 0.001);
    }
}

// Test case 4
TEST(TensorTest, FloatingPointMultiplicationWithScalar)
{
    auto t5 = std::make_shared<tensor<float>>(std::vector<int>{5, 3, 2});
    t5->data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30};

    auto res = 1.5f * t5;
    std::vector<float> expected = {1.5000, 3.0000,
                                   4.5000, 6.0000,
                                   7.5000, 9.0000,

                                   10.5000, 12.0000,
                                   13.5000, 15.0000,
                                   16.5000, 18.0000,

                                   19.5000, 21.0000,
                                   22.5000, 24.0000,
                                   25.5000, 27.0000,

                                   28.5000, 30.0000,
                                   31.5000, 33.0000,
                                   34.5000, 36.0000,

                                   37.5000, 39.0000,
                                   40.5000, 42.0000,
                                   43.5000, 45.0000};

    for (int i = 0; i < res->total; i++)
    {
        EXPECT_NEAR(res->data[i], expected[i], 0.001);
    }
}

// Test cases where the shape mismatch triggers an assertion failure
TEST(TensorAssertionFailureTest, TensorMultiplicationShapeMismatch)
{
    auto t1 = std::make_shared<tensor<int>>(std::vector<int>{2, 3});
    auto t2 = std::make_shared<tensor<int>>(std::vector<int>{3, 2});
    t1->data = {1, 2, 3, 4, 5, 6};
    t2->data = {1, 2, 3, 4, 5, 6};

    EXPECT_THROW({ auto res = t1 * t2; }, std::runtime_error);
}

TEST(TensorAssertionFailureTest, TensorMultiplicationWithScalarShapeMismatch)
{
    auto t1 = std::make_shared<tensor<float>>(std::vector<int>{3});
    auto t2 = std::make_shared<tensor<float>>(std::vector<int>{1, 3});
    t1->data = {1.0f, 2.0f, 3.0f};
    t2->data = {1.0f, 2.0f, 3.0f};

    EXPECT_THROW({ auto res = t1 * t2; }, std::runtime_error);
}