#include "synaptic.hpp"
#include <gtest/gtest.h>


TEST(TensorTest, TensorSoftmax)
{
    auto t4 = std::make_shared<synaptic::tensor<float>>(std::vector<int>{5, 3, 2});
    t4->data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30};

    synaptic::connections::softmax<float> softmax = synaptic::connections::softmax<float>();
    auto res = softmax.forward(t4);
    //std::cout << *res << std::endl;
    std::vector<float> expected = std::vector<float>{1.6079e-13, 4.3707e-13, 1.1881e-12, 3.2296e-12, 8.7789e-12, 2.3863e-11,
        6.4867e-11, 1.7633e-10, 4.7931e-10, 1.3029e-09, 3.5416e-09, 9.6272e-09,
        2.6169e-08, 7.1136e-08, 1.9337e-07, 5.2563e-07, 1.4288e-06, 3.8839e-06,
        1.0557e-05, 2.8698e-05, 7.8010e-05, 2.1205e-04, 5.7642e-04, 1.5669e-03,
        4.2592e-03, 1.1578e-02, 3.1471e-02, 8.5548e-02, 2.3254e-01, 6.3212e-01};

    for (int i = 0; i < res->data.size(); i++)
    {
        EXPECT_NEAR(res->data[i], expected[i],0.001);
    }
}

/* TEST(TensorTest, TensorsoftmaxBackpropCheck)
{
    auto t4 = std::make_shared<synaptic::tensor<float>>(std::vector<int>{2, 2});
    t4->data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30};

    synaptic::connections::softmax<float> softmax = synaptic::connections::softmax<float>();
    auto res = softmax.forward(t4);
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
} */


