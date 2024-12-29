#include "../../include/synaptic.hpp"
#include <vector>
#include <memory>
#include <gtest/gtest.h> // Include the GoogleTest header
#include <stdexcept>

using namespace synaptic;
// Test case 1
TEST(TensorTest, CrossEntropyOfTwoTensors)
{
    auto t1 = std::make_shared<tensor<float>>(std::vector<int>{5,4});
    auto t2 = std::make_shared<tensor<float>>(std::vector<int>{5,4});
    t1->data = {1.1510, -0.6861,  1.6330, -1.0105,
        -0.1439, -0.4433, -0.1430, -1.2572,
        -1.3980,  0.8178,  0.0507, -0.1254,
         0.3683, -0.0996, -0.0407, -0.2591,
         0.2070,  1.9050, -0.4315, -0.2798};
    t2->data = {0., 0., 1., 0.,
        0., 0., 1., 0.,
        0., 0., 1., 0.,
        0., 0., 1., 0.,
        0., 0., 1., 0.};

    auto cross_entropy_loss = synaptic::loss_fn::cross_entropy_loss<float>();
    //std::cout <<"inside" << std::endl;
    auto res = cross_entropy_loss.forward(t1,t2);
    std::vector<float> expected = {1.4516};
    //std::cout<< res->total <<std::endl;
    for (int i = 0; i < res->total; i++) {
        EXPECT_NEAR(res->data[i], expected[i],0.0001);

    }
}

TEST(TensorTest, CrossEntropyOfTwoTensorsBackpropCheck)
{
    auto t1 = std::make_shared<tensor<float>>(std::vector<int>{5,4});
    auto t2 = std::make_shared<tensor<float>>(std::vector<int>{5,4});
    t1->data = {1.1510, -0.6861,  1.6330, -1.0105,
        -0.1439, -0.4433, -0.1430, -1.2572,
        -1.3980,  0.8178,  0.0507, -0.1254,
         0.3683, -0.0996, -0.0407, -0.2591,
         0.2070,  1.9050, -0.4315, -0.2798};
    t2->data = {0., 0., 1., 0.,
        0., 0., 1., 0.,
        0., 0., 1., 0.,
        0., 0., 1., 0.,
        0., 0., 1., 0.};

    auto cross_entropy_loss = synaptic::loss_fn::cross_entropy_loss<float>();
    //std::cout <<"inside" << std::endl;
    auto res = cross_entropy_loss.forward(t1,t2);
    std::vector<float> expected = {0.0691,  0.0110, -0.0881,  0.0080,
        0.0651,  0.0483, -0.1348,  0.0214,
        0.0111,  0.1019, -0.1527,  0.0397,
        0.0708,  0.0443, -0.1530,  0.0378,
        0.0263,  0.1437, -0.1861,  0.0162};
    //std::cout<< res->total <<std::endl;
    res->backprop();
    for (int i = 0; i < res->total; i++) {
        EXPECT_NEAR(t1->grad[i], expected[i],0.0001);

    }
}


// Test cases where the shape mismatch triggers an assertion failure
TEST(TensorAssertionFailureTest, TensorCrossEntropyShapeMismatch) {
    auto t1 = std::make_shared<tensor<float>>(std::vector<int>{2, 3});
    auto t2 = std::make_shared<tensor<float>>(std::vector<int>{3, 2});
    t1->data = {1, 2, 3, 4, 5, 6};
    t2->data = {1, 2, 3, 4, 5, 6};

    auto cross_entropy_loss = synaptic::loss_fn::cross_entropy_loss<float>();
    
    EXPECT_THROW({
        std::shared_ptr<tensor<float>> res = cross_entropy_loss.forward(t1,t2);
    },std::runtime_error);  
}

TEST(TensorAssertionFailureTest, TensorCrossEntropyWithScalarShapeMismatch) {
    auto t1 = std::make_shared<tensor<float>>(std::vector<int>{3});
    auto t2 = std::make_shared<tensor<float>>(std::vector<int>{1, 3});
    t1->data = {1.0f, 2.0f, 3.0f};
    t2->data = {1.0f, 2.0f, 3.0f};

    
    auto cross_entropy_loss = synaptic::loss_fn::cross_entropy_loss<float>();
    
    EXPECT_THROW({
        auto res = cross_entropy_loss.forward(t1,t2);
    },std::runtime_error);
}