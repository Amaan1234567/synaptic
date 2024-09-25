#include "../../include/synaptic.hpp"
#include <gtest/gtest.h>


TEST(TensorTest, TensorSigmoid)
{
    auto t4 = std::make_shared<synaptic::tensor<float>>(std::vector<int>{5, 3, 2});
    t4->data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30};

    synaptic::connections::sigmoid<float> sigmoid = synaptic::connections::sigmoid<float>();
    auto res = sigmoid.forward(t4);
    //std::cout << *res << std::endl;
    std::vector<float> expected = std::vector<float>{0.7311, 0.8808, 0.9526, 0.9820, 0.9933, 0.9975, 0.9991, 0.9997, 0.9999,
        1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
        1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
        1.0000, 1.0000, 1.0000};

    for (int i = 0; i < res->data.size(); i++)
    {
        EXPECT_NEAR(res->data[i], expected[i],0.001);
    }
}

TEST(TensorTest, TensorSigmoidWithSlopeParam)
{
    auto t4 = std::make_shared<synaptic::tensor<float>>(std::vector<int>{5, 3, 2});
    t4->data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30};

    synaptic::connections::sigmoid<float> sigmoid = synaptic::connections::sigmoid<float>(2.0f);
    auto res = sigmoid.forward(t4);
    //std::cout << *res << std::endl;
    std::vector<float> expected = std::vector<float>{0.8808, 0.9820, 0.9975, 0.9997, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
        1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
        1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
        1.0000, 1.0000, 1.0000};

    for (int i = 0; i < res->dims.size(); i++)
    {
        EXPECT_NEAR(res->data[i], expected[i],0.001);
    }
}

TEST(TensorTest, TensorSigmoidBackpropCheck)
{
    auto t4 = std::make_shared<synaptic::tensor<float>>(std::vector<int>{2, 2});
    t4->data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30};

    synaptic::connections::sigmoid<float> sigmoid = synaptic::connections::sigmoid<float>();
    auto res = sigmoid.forward(t4);
    //std::cout << *res << std::endl;
    std::vector<float> expected = std::vector<float>{1.9661e-01, 1.0499e-01, 4.5177e-02, 1.7663e-02, 6.6480e-03, 2.4665e-03,
        9.1017e-04, 3.3522e-04, 1.2337e-04, 4.5417e-05, 1.6689e-05, 6.1988e-06,
        2.2650e-06, 8.3446e-07, 3.5763e-07, 1.1921e-07, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00};

    res->backprop();
    for (int i = 0; i < res->total; i++)
    {
        EXPECT_NEAR(t4->grad[i], expected[i],0.001);
    }
}


TEST(TensorTest, TensorSigmoidBackpropCheckWithSlopeParam)
{
    auto t4 = std::make_shared<synaptic::tensor<float>>(std::vector<int>{2, 2});
    t4->data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30};

    synaptic::connections::sigmoid<float> sigmoid = synaptic::connections::sigmoid<float>(2.0f);
    auto res = sigmoid.forward(t4);
    //std::cout << *res << std::endl;
    std::vector<float> expected = std::vector<float>{2.0999e-01, 3.5325e-02, 4.9330e-03, 6.7048e-04, 9.0792e-05, 1.2288e-05,
        1.6631e-06, 2.2507e-07, 3.0460e-08, 4.1223e-09, 5.5789e-10, 7.5503e-11,
        1.0218e-11, 1.3829e-12, 1.8715e-13, 2.5328e-14, 3.4278e-15, 4.6390e-16,
        6.2783e-17, 8.4967e-18, 1.1499e-18, 1.5562e-19, 2.1061e-20, 2.8503e-21,
        3.8575e-22, 5.2206e-23, 7.0653e-24, 9.5618e-25, 1.2940e-25, 1.7513e-26};

    res->backprop();
    for (int i = 0; i < res->total; i++)
    {
        EXPECT_NEAR(t4->grad[i], expected[i],0.001);
    }
}