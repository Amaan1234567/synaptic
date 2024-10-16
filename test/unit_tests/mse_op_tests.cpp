#include "../../include/synaptic.hpp"
#include <vector>
#include <memory>
#include <gtest/gtest.h> // Include the GoogleTest header
#include <stdexcept>

using namespace synaptic;
// Test case 1
TEST(TensorTest, MseOfTwoTensors)
{
    auto t1 = std::make_shared<tensor<float>>(std::vector<int>{5,4});
    auto t2 = std::make_shared<tensor<float>>(std::vector<int>{5,4});
    t1->data = {2.0153,  1.5694,  1.1120,  0.5215,
        -0.0261,  1.1867,  0.4831, -1.4634,
        -1.4566,  0.1550,  0.1747, -0.7770,
        -0.5263, -0.8394, -1.7548, -0.6452,
        -1.1722, -0.7855, -0.5609,  0.2995};
    t2->data = {1., 1., 1., 1.,
        1., 1., 1., 1.,
        1., 1., 1., 1.,
        1., 1., 1., 1.,
        1., 1., 1., 1.};

    auto mse = synaptic::loss_fn::mse<float>();
    //std::cout <<"inside" << std::endl;
    auto res = mse.forward(t1,t2);
    std::vector<float> expected = {2.3225};
    //std::cout<< res->total <<std::endl;
    for (int i = 0; i < res->total; i++) {
        EXPECT_NEAR(res->data[i], expected[i],0.0001);

    }
}

TEST(TensorTest, MseOfTwoTensorsBackpropCheck)
{
    auto t1 = std::make_shared<tensor<float>>(std::vector<int>{5,4});
    auto t2 = std::make_shared<tensor<float>>(std::vector<int>{5,4});
    t1->data = {2.0153,  1.5694,  1.1120,  0.5215,
        -0.0261,  1.1867,  0.4831, -1.4634,
        -1.4566,  0.1550,  0.1747, -0.7770,
        -0.5263, -0.8394, -1.7548, -0.6452,
        -1.1722, -0.7855, -0.5609,  0.2995};
    t2->data = {1., 1., 1., 1.,
        1., 1., 1., 1.,
        1., 1., 1., 1.,
        1., 1., 1., 1.,
        1., 1., 1., 1.};

    auto mse = synaptic::loss_fn::mse<float>();

    auto res = mse.forward(t1,t2);

    res->backprop();

    std::vector<float> expected1 = { 0.1015,  0.0569,  0.0112, -0.0478,
        -0.1026,  0.0187, -0.0517, -0.2463,
        -0.2457, -0.0845, -0.0825, -0.1777,
        -0.1526, -0.1839, -0.2755, -0.1645,
        -0.2172, -0.1786, -0.1561, -0.0700};

    std::vector<float> expected2 = {-0.1015, -0.0569, -0.0112,  0.0478,
        0.1026, -0.0187,  0.0517,  0.2463,
        0.2457,  0.0845,  0.0825,  0.1777,
        0.1526,  0.1839,  0.2755,  0.1645,
        0.2172,  0.1786,  0.1561,  0.0700};

    for (int i = 0; i < res->total; i++) {
        EXPECT_NEAR(t1->grad[i], expected1[i],0.0001);
        EXPECT_NEAR(t2->grad[i], expected2[i],0.0001);
    }
}


// Test cases where the shape mismatch triggers an assertion failure
TEST(TensorAssertionFailureTest, TensorMseShapeMismatch) {
    auto t1 = std::make_shared<tensor<float>>(std::vector<int>{2, 3});
    auto t2 = std::make_shared<tensor<float>>(std::vector<int>{3, 2});
    t1->data = {1, 2, 3, 4, 5, 6};
    t2->data = {1, 2, 3, 4, 5, 6};

    auto mse = synaptic::loss_fn::mse<float>();
    
    EXPECT_THROW({
        std::shared_ptr<tensor<float>> res = mse.forward(t1,t2);
    },std::runtime_error);  
}

TEST(TensorAssertionFailureTest, TensorMseWithScalarShapeMismatch) {
    auto t1 = std::make_shared<tensor<float>>(std::vector<int>{3});
    auto t2 = std::make_shared<tensor<float>>(std::vector<int>{1, 3});
    t1->data = {1.0f, 2.0f, 3.0f};
    t2->data = {1.0f, 2.0f, 3.0f};

    
    auto mse = synaptic::loss_fn::mse<float>();
    
    EXPECT_THROW({
        auto res = mse.forward(t1,t2);
    },std::runtime_error);
}