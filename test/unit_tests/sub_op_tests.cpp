#include "../../include/tensor.hpp"
#include <vector>
#include <memory>
#include <gtest/gtest.h> // Include the GoogleTest header
#include <stdexcept>

// Test case 1
TEST(TensorTest, SubtractionOfTwoTensors)
{
    auto t1 = std::make_shared<tensor<float>>(std::vector<int>{2});
    auto t2 = std::make_shared<tensor<float>>(std::vector<int>{2});
    t1->data = {1.0, 3.0};
    t2->data = {1.0, 4.0};

    auto res = t1 - t2;
    std::vector<float> expected = {0.0, -1.0};

    for (int i = 0; i < res->total; i++) {
        EXPECT_FLOAT_EQ(res->data[i], expected[i]);

    }
}

TEST(TensorTest, SubtractionOfTwoTensorsBackpropCheck)
{
    auto t1 = std::make_shared<tensor<float>>(std::vector<int>{2});
    auto t2 = std::make_shared<tensor<float>>(std::vector<int>{2});
    t1->data = {1.0, 3.0};
    t2->data = {1.0, 4.0};

    auto res = t1 - t2;
    std::vector<float> expected1 = {1.0, 1.0};
    std::vector<float> expected2 = {-1.0, -1.0};

    res->backprop();
    for (int i = 0; i < res->total; i++) {
        EXPECT_FLOAT_EQ(t1->grad[i], expected1[i]);
        EXPECT_FLOAT_EQ(t2->grad[i], expected2[i]);
        
    }
}

// Test case 2
TEST(TensorTest, SubtractionWithScalar)
{
    auto t3 = std::make_shared<tensor<int>>(std::vector<int>{5, 3, 2});
    t3->data = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
    

    auto res = t3 - 1;
    std::vector<int> expected = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30};

    for (int i = 0; i < res->total; i++) {
        EXPECT_EQ(res->data[i], expected[i]);  // Using EXPECT_EQ for integer comparison
    }
}

// Test case 3
TEST(TensorTest, ScalarSubtractionToTensor)
{
    auto t4 = std::make_shared<tensor<int>>(std::vector<int>{5, 3, 2});
    t4->data = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};

    auto res = 1 - t4;
    std::vector<int> expected = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30};
    

    for (int i = 0; i < res->total; i++) {
        EXPECT_EQ(res->data[i], expected[i]);
    }
}

// Test case 4
TEST(TensorTest, FloatingPointSubtractionWithScalar)
{
    auto t5 = std::make_shared<tensor<float>>(std::vector<int>{5, 3, 2});
    t5->data = {2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5, 28.5, 29.5, 30.5, 31.5};
    
    auto res = 1.5f - t5;
    std::vector<float> expected = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30};

    for (int i = 0; i < res->total; i++) {
        EXPECT_FLOAT_EQ(res->data[i], expected[i]);
    }
}

// Test cases where the shape mismatch triggers an assertion failure
TEST(TensorAssertionFailureTest, TensorSubtractionShapeMismatch) {
    auto t1 = std::make_shared<tensor<int>>(std::vector<int>{2, 3});
    auto t2 = std::make_shared<tensor<int>>(std::vector<int>{3, 2});
    t1->data = {1, 2, 3, 4, 5, 6};
    t2->data = {1, 2, 3, 4, 5, 6};

    
    EXPECT_THROW({
        auto res = t1 - t2;
    },std::runtime_error);  
}

TEST(TensorAssertionFailureTest, TensorSubtractionWithScalarShapeMismatch) {
    auto t1 = std::make_shared<tensor<float>>(std::vector<int>{3});
    auto t2 = std::make_shared<tensor<float>>(std::vector<int>{1, 3});
    t1->data = {1.0f, 2.0f, 3.0f};
    t2->data = {1.0f, 2.0f, 3.0f};

    
    EXPECT_THROW({
        auto res = t1 - t2;
    },std::runtime_error);
}