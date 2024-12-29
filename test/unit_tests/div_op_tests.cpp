#include "../../include/synaptic.hpp"
#include <vector>
#include <memory>
#include <gtest/gtest.h>
#include <stdexcept>

using namespace synaptic;

TEST(TensorTest, DivisionOfTwoTensors)
{
    auto t1 = std::make_shared<tensor<float>>(std::vector<int>{2});
    auto t2 = std::make_shared<tensor<float>>(std::vector<int>{2});
    t1->data = {1.0, 3.0};
    t2->data = {1.0, 4.0};

    auto res = t1 / t2;
    std::vector<float> expected = {1.0, 0.75};

    for (int i = 0; i < res->total; i++) {
        EXPECT_NEAR(res->data[i], expected[i],0.001); 
    }
}

TEST(TensorTest, DivisionOfTwoTensorsBackpropCheck)
{
    auto t1 = std::make_shared<tensor<float>>(std::vector<int>{3});
    auto t2 = std::make_shared<tensor<float>>(std::vector<int>{3});
    auto t3 = std::make_shared<tensor<float>>(std::vector<int>{3});
    t1->data = {1, 2, 3};
    t2->data = {4, 5, 6};
    t3->data = {1, 1, 1};
    auto c = t1 / t2;
    auto r = t3+c;
    r->backprop();
    std::vector<float> expected1 = {0.2500, 0.2000, 0.1667};
    std::vector<float> expected2 = {-0.0625, -0.0800, -0.0833};

    for (int i = 0; i < t1->total; i++) {
        EXPECT_NEAR(t1->grad[i], expected1[i],0.001);
        EXPECT_NEAR(t2->grad[i], expected2[i],0.001); 
    }
}

TEST(TensorTest, DivisionOfTwoTensorsZeroDivision)
{
    auto t1 = std::make_shared<tensor<float>>(std::vector<int>{2});
    auto t2 = std::make_shared<tensor<float>>(std::vector<int>{2});
    t1->data = {1.0, 2.0};
    t2->data = {0.0, 4.0};

    auto res = t1 / t2;
    std::vector<float> expected = {std::numeric_limits<float>::infinity(), 0.5};

    for (int i = 0; i < res->total; i++) {
        EXPECT_FLOAT_EQ(res->data[i], expected[i]); 
    }
}

// Test case 2
TEST(TensorTest, DivisionWithScalar)
{
    auto t3 = std::make_shared<tensor<int>>(std::vector<int>{5, 3, 2});
    t3->data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30};

    auto res = t3 / 1;
    std::vector<int> expected = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30};

    for (int i = 0; i < res->total; i++) {
        EXPECT_EQ(res->data[i], expected[i]);  // Using EXPECT_EQ for integer comparison
    }
}

TEST(TensorTest, DivisionWithScalarInt)
{
    auto t3 = std::make_shared<tensor<int>>(std::vector<int>{5, 3, 2});
    t3->data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30};

    auto res = 1/t3 ;
    std::vector<int> expected = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0};

    for (int i = 0; i < res->total; i++) {
        EXPECT_EQ(res->data[i], expected[i]);  // Using EXPECT_EQ for integer comparison
    }
}

// Test case 3
TEST(TensorTest, ScalarDivisionToTensor)
{
    auto t4 = std::make_shared<tensor<float>>(std::vector<int>{5, 3, 2});
    t4->data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30};

    auto res = 1.0f / t4;
    std::vector<float> expected = {1.0000, 0.5000, 0.3333, 0.2500, 0.2000, 0.1667, 0.1429, 0.1250, 0.1111,
        0.1000, 0.0909, 0.0833, 0.0769, 0.0714, 0.0667, 0.0625, 0.0588, 0.0556,
        0.0526, 0.0500, 0.0476, 0.0455, 0.0435, 0.0417, 0.0400, 0.0385, 0.0370,
        0.0357, 0.0345, 0.0333};

    for (int i = 0; i < res->total; i++) {
        EXPECT_NEAR(res->data[i], expected[i],0.001);
    }
}

// Test case 4
TEST(TensorTest, FloatingPointDivisionWithScalar)
{
    auto t5 = std::make_shared<tensor<float>>(std::vector<int>{5, 3, 2});
    t5->data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30};

    auto res = 1.5f / t5;
    std::vector<float> expected = {1.5000, 0.7500, 0.5000, 0.3750, 0.3000, 0.2500, 0.2143, 0.1875, 0.1667,
        0.1500, 0.1364, 0.1250, 0.1154, 0.1071, 0.1000, 0.0938, 0.0882, 0.0833,
        0.0789, 0.0750, 0.0714, 0.0682, 0.0652, 0.0625, 0.0600, 0.0577, 0.0556,
        0.0536, 0.0517, 0.0500};

    for (int i = 0; i < res->total; i++) {
        EXPECT_NEAR(res->data[i], expected[i],0.001);
    }
}

// Test cases where the shape mismatch triggers an assertion failure
TEST(TensorAssertionFailureTest, TensorDivisionShapeMismatch) {
    auto t1 = std::make_shared<tensor<int>>(std::vector<int>{2, 3});
    auto t2 = std::make_shared<tensor<int>>(std::vector<int>{3, 2});
    t1->data = {1, 2, 3, 4, 5, 6};
    t2->data = {1, 2, 3, 4, 5, 6};

    
    EXPECT_THROW({
        auto res = t1 / t2;
    },std::runtime_error);  
}

TEST(TensorAssertionFailureTest, TensorDivisionWithScalarShapeMismatch) {
    auto t1 = std::make_shared<tensor<float>>(std::vector<int>{3});
    auto t2 = std::make_shared<tensor<float>>(std::vector<int>{1, 3});
    t1->data = {1.0f, 2.0f, 3.0f};
    t2->data = {1.0f, 2.0f, 3.0f};

    
    EXPECT_THROW({
        auto res = t1 / t2;
    },std::runtime_error);
}