#include "../../include/synaptic.hpp"
#include <vector>
#include <memory>
#include <gtest/gtest.h> // Include the GoogleTest header
#include <stdexcept>

using namespace synaptic;

TEST(TensorTest, TensorLog)
{
    auto t1 = std::make_shared<tensor<float>>(std::vector<int>{10});
    t1->data = {9., 4., 6., 7., 7., 3., 8., 4., 2., 9.};

    auto res = tensor<float>::log(t1);
    std::vector<float> expected = {2.1972, 1.3863, 1.7918, 1.9459, 1.9459, 1.0986, 2.0794, 1.3863, 0.6931,
        2.1972};

    for (int i = 0; i < res->total; i++) {
        EXPECT_NEAR(res->data[i], expected[i],0.0001);
    }
}

TEST(TensorTest, TensorLogBase10)
{
    auto t1 = std::make_shared<tensor<float>>(std::vector<int>{10});
    t1->data = {9., 4., 6., 7., 7., 3., 8., 4., 2., 9.};

    auto res = tensor<float>::log(t1,10);
    std::vector<float> expected = {0.9542, 0.6021, 0.7782, 0.8451, 0.8451, 0.4771, 0.9031, 0.6021, 0.3010,
        0.9542};

    for (int i = 0; i < res->total; i++) {
        EXPECT_NEAR(res->data[i], expected[i],0.001);

    }
}


TEST(TensorTest, TensorLogCustomBase)
{
    auto t1 = std::make_shared<tensor<float>>(std::vector<int>{10});
    t1->data = {6., 8., 7., 9., 5., 9., 4., 7., 9., 5.};

    auto res = tensor<float>::log(t1,42.167);
    std::vector<float> expected = {0.4789, 0.5558, 0.5201, 0.5872, 0.4301, 0.5872, 0.3705, 0.5201, 0.5872,
        0.4301};

    for (int i = 0; i < res->total; i++) {
        EXPECT_NEAR(res->data[i], expected[i],0.0001);

    }
}


TEST(TensorTest, TensorLogBackpropCheck)
{
    auto t1 = std::make_shared<tensor<float>>(std::vector<int>{10});
    t1->data = {3.,  6.,  6., 10.,  9., 10.,  9.,  3.,  8.,  4.};

    auto res = tensor<float>::log(t1);
    std::vector<float> expected1 = {0.3333, 0.1667, 0.1667, 0.1000, 0.1111, 0.1000, 0.1111, 0.3333, 0.1250,
        0.2500};
    res->backprop();
    //std::cout<< *t1 <<std::endl;
    for (int i = 0; i < res->total; i++) {
        EXPECT_NEAR(t1->grad[i], expected1[i],0.0001);
    }
}

TEST(TensorTest, TensorLogBackpropCheckCustomBase)
{
    auto t1 = std::make_shared<tensor<float>>(std::vector<int>{10});
    t1->data = {1., 2., 7., 1., 6., 7., 2., 5., 6., 9.};

    auto res = tensor<float>::log(t1,42.167);
    std::vector<float> expected1 = {0.2673, 0.1336, 0.0382, 0.2673, 0.0445, 0.0382, 0.1336, 0.0535, 0.0445,
        0.0297};
    res->backprop();
    std::cout << *res << std::endl;
    std::cout<< *t1 <<std::endl;
    for (int i = 0; i < res->total; i++) {
        EXPECT_NEAR(t1->grad[i], expected1[i],0.0001);
    }
}

TEST(TensorAssertionFailureTest,TensorLogNegBaseCheck)
{
    auto t1 = std::make_shared<tensor<float>>(std::vector<int>{10});
    t1->data = {1., 2., 7., 1., 6., 7., 2., 5., 6., 9.};

    
    EXPECT_THROW({ auto res = tensor<float>::log(t1,-10); }, std::runtime_error);
}

TEST(TensorAssertionFailureTest,TensorLogZeroBaseCheck)
{
    auto t1 = std::make_shared<tensor<float>>(std::vector<int>{10});
    t1->data = {1., 2., 7., 1., 6., 7., 2., 5., 6., 9.};

    
    EXPECT_THROW({ auto res = tensor<float>::log(t1,0); }, std::runtime_error);
}