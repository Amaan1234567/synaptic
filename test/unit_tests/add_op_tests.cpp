#include "../../include/tensor.hpp"
#include <vector>
#include <memory>
#include <gtest/gtest.h> // Include the GoogleTest header
#include <stdexcept>

// Test case 1
TEST(TensorTest, AdditionOfTwoTensors)
{
    auto t1 = std::make_shared<tensor<float>>(std::vector<int>{2});
    auto t2 = std::make_shared<tensor<float>>(std::vector<int>{2});
    t1->data = {1.0, 3.0};
    t2->data = {1.0, 4.0};

    auto res = t1 + t2;
    std::vector<float> expected = {2.0, 7.0};

    for (int i = 0; i < res->total; i++) {
        EXPECT_FLOAT_EQ(res->data[i], expected[i]);

    }
}

TEST(TensorTest, AdditionOfTwoTensorsBackpropCheck)
{
    auto t1 = std::make_shared<tensor<float>>(std::vector<int>{2});
    auto t2 = std::make_shared<tensor<float>>(std::vector<int>{2});
    t1->data = {1.0, 3.0};
    t2->data = {1.0, 4.0};

    auto res = t1 + t2;
    std::vector<float> expected = {1.0, 1.0};

    res->backprop();
    for (int i = 0; i < res->total; i++) {
        EXPECT_FLOAT_EQ(t1->grad[i], expected[i]);
        EXPECT_FLOAT_EQ(t2->grad[i], expected[i]);
        
    }
}

// Test case 2
TEST(TensorTest, AdditionWithScalar)
{
    auto t3 = std::make_shared<tensor<int>>(std::vector<int>{5, 3, 2});
    t3->data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30};

    auto res = t3 + 1;
    std::vector<int> expected = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};

    for (int i = 0; i < res->total; i++) {
        EXPECT_EQ(res->data[i], expected[i]);  // Using EXPECT_EQ for integer comparison
    }
}

// Test case 3
TEST(TensorTest, ScalarAdditionToTensor)
{
    auto t4 = std::make_shared<tensor<int>>(std::vector<int>{5, 3, 2});
    t4->data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30};

    auto res = 1 + t4;
    std::vector<int> expected = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};

    for (int i = 0; i < res->total; i++) {
        EXPECT_EQ(res->data[i], expected[i]);
    }
}

// Test case 4
TEST(TensorTest, FloatingPointAdditionWithScalar)
{
    auto t5 = std::make_shared<tensor<float>>(std::vector<int>{5, 3, 2});
    t5->data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30};

    auto res = 1.5f + t5;
    std::vector<float> expected = {2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5, 28.5, 29.5, 30.5, 31.5};

    for (int i = 0; i < res->total; i++) {
        EXPECT_FLOAT_EQ(res->data[i], expected[i]);
    }
}

// Test cases where the shape mismatch triggers an assertion failure
TEST(TensorAssertionFailureTest, TensorAdditionShapeMismatch) {
    auto t1 = std::make_shared<tensor<int>>(std::vector<int>{2, 3});
    auto t2 = std::make_shared<tensor<int>>(std::vector<int>{3, 2});
    t1->data = {1, 2, 3, 4, 5, 6};
    t2->data = {1, 2, 3, 4, 5, 6};

    
    EXPECT_THROW({
        auto res = t1 + t2;
    },std::runtime_error);  
}

TEST(TensorAssertionFailureTest, TensorAdditionWithScalarShapeMismatch) {
    auto t1 = std::make_shared<tensor<float>>(std::vector<int>{3});
    auto t2 = std::make_shared<tensor<float>>(std::vector<int>{1, 3});
    t1->data = {1.0f, 2.0f, 3.0f};
    t2->data = {1.0f, 2.0f, 3.0f};

    
    EXPECT_THROW({
        auto res = t1 + t2;
    },std::runtime_error);
}