#include "synaptic.hpp"
#include <gtest/gtest.h>


TEST(TensorTest, TensorSoftmax)
{
    auto t4 = std::make_shared<synaptic::tensor<float>>(std::vector<int>{30});
    t4->data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30};

    synaptic::connections::softmax<float> softmax = synaptic::connections::softmax<float>();
    auto res = softmax.forward(t4);
    std::cout << *res << std::endl;
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

TEST(TensorTest, TensorSoftmaxBackpropCheck)
{


    auto t4 = std::make_shared<synaptic::tensor<float>>(std::vector<int>{30});
    t4->data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30};

    synaptic::connections::softmax<float> softmax = synaptic::connections::softmax<float>();
    auto res = softmax.forward(t4);
    std::cout << "inside" << std::endl;
    
    std::vector<float> expected = std::vector<float>{-1.9168e-20, -5.2103e-20, -1.4163e-19, -3.8499e-19, -1.0465e-18,
        -2.8447e-18, -7.7328e-18, -2.1020e-17, -5.7138e-17, -1.5532e-16,
        -4.2220e-16, -1.1476e-15, -3.1196e-15, -8.4800e-15, -2.3051e-14,
        -6.2660e-14, -1.7033e-13, -4.6299e-13, -1.2586e-12, -3.4211e-12,
        -9.2995e-12, -2.5279e-11, -6.8715e-11, -1.8679e-10, -5.0774e-10,
        -1.3802e-09, -3.7517e-09, -1.0198e-08, -2.7721e-08, -7.5355e-08};

    res->backprop();
    std::cout << *t4 << std::endl;
    for (int i = 0; i < res->total; i++)
    {
        EXPECT_NEAR(t4->grad[i], expected[i],0.00001);
    }
}


