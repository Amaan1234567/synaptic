#include "../../include/synaptic.hpp"
#include <vector>
#include <memory>
#include <gtest/gtest.h> // Include the GoogleTest header
#include <stdexcept>

using namespace synaptic;
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

TEST(TensorTest, AdditionOfTwoTensorsWithDifferentBatchSizes)
{
    auto t1 = std::make_shared<tensor<float>>(std::vector<int>{16,5});
    auto t2 = std::make_shared<tensor<float>>(std::vector<int>{1,5});
    t1->data = {3.,  2.,  1.,  9.,  3.,
         3.,  6.,  5.,  4.,  6.,
         6.,  6.,  9.,  2.,  9.,
         8.,  9.,  5.,  1.,  6.,
         7.,  7.,  9.,  5.,  8.,
         5.,  2.,  6.,  4.,  9.,
         9.,  5.,  6.,  6.,  2.,
        10.,  7.,  2.,  8.,  8.,
         5.,  9., 10.,  6.,  7.,
         5.,  6.,  6.,  5.,  4.,
         6.,  4.,  9.,  3.,  7.,
         7.,  5.,  6.,  7., 10.,
         9.,  6.,  6.,  8., 10.,
         2.,  6., 10.,  9.,  4.,
         8.,  9.,  5.,  1.,  7.,
         5., 10.,  2., 10., 10.};
    t2->data = {10.,  5.,  5.,  5.,  9.};

    auto res = t1 + t2;
    std::vector<float> expected = {13.,  7.,  6., 14., 12.,
        13., 11., 10.,  9., 15.,
        16., 11., 14.,  7., 18.,
        18., 14., 10.,  6., 15.,
        17., 12., 14., 10., 17.,
        15.,  7., 11.,  9., 18.,
        19., 10., 11., 11., 11.,
        20., 12.,  7., 13., 17.,
        15., 14., 15., 11., 16.,
        15., 11., 11., 10., 13.,
        16.,  9., 14.,  8., 16.,
        17., 10., 11., 12., 19.,
        19., 11., 11., 13., 19.,
        12., 11., 15., 14., 13.,
        18., 14., 10.,  6., 16.,
        15., 15.,  7., 15., 19.};

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

TEST(TensorTest, AdditionOfTwoTensorsWithDifferentBatchSizeBackpropCheck)
{
    auto t1 = std::make_shared<tensor<float>>(std::vector<int>{16,5});
    auto t2 = std::make_shared<tensor<float>>(std::vector<int>{1,5});
    t1->data = {3.,  2.,  1.,  9.,  3.,
         3.,  6.,  5.,  4.,  6.,
         6.,  6.,  9.,  2.,  9.,
         8.,  9.,  5.,  1.,  6.,
         7.,  7.,  9.,  5.,  8.,
         5.,  2.,  6.,  4.,  9.,
         9.,  5.,  6.,  6.,  2.,
        10.,  7.,  2.,  8.,  8.,
         5.,  9., 10.,  6.,  7.,
         5.,  6.,  6.,  5.,  4.,
         6.,  4.,  9.,  3.,  7.,
         7.,  5.,  6.,  7., 10.,
         9.,  6.,  6.,  8., 10.,
         2.,  6., 10.,  9.,  4.,
         8.,  9.,  5.,  1.,  7.,
         5., 10.,  2., 10., 10.};
    t2->data = {10.,  5.,  5.,  5.,  9.};

    auto res = t1 + t2;
    std::vector<float> expected1(t1->total,1.0);
    std::vector<float> expected2(t2->total,int(t1->dims[0]));
    res->backprop();
    for (size_t i = 0; i < res->total; i++) {
        EXPECT_FLOAT_EQ(t1->grad[i], expected1[i]);  
    }
    for(size_t i = 0;i<t2->total;i++)
    {
        EXPECT_FLOAT_EQ(t2->grad[i], expected2[i]);
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