#include "../../include/synaptic.hpp"
#include <gtest/gtest.h>


TEST(TensorTest, TensorTanh)
{
    auto t4 = std::make_shared<synaptic::tensor<float>>(std::vector<int>{5, 3, 2});
    t4->data = { 1.1438, -0.4469, -0.4013,  0.2292, -1.0918,  0.2629,  1.1213, -0.6684,
        -0.4937, -0.1513,  1.0481,  0.6175,  0.2499,  0.0127, -1.5668, -0.6067,
        -0.5968, -0.2543, -0.7310,  0.0811,  0.3471, -1.4205, -0.0382, -1.2393,
         0.4111, -0.6613,  0.7922,  0.5549,  0.7943,  1.7873};

    synaptic::connections::tanh<float> tanh = synaptic::connections::tanh<float>();
    auto res = tanh.forward(t4);
    //std::cout << *res << std::endl;
    std::vector<float> expected = std::vector<float>{ 0.8157, -0.4193, -0.3810,  0.2252, -0.7975,  0.2571,  0.8080, -0.5839,
        -0.4571, -0.1501,  0.7811,  0.5494,  0.2448,  0.0127, -0.9165, -0.5418,
        -0.5348, -0.2490, -0.6237,  0.0809,  0.3338, -0.8897, -0.0382, -0.8452,
         0.3894, -0.5792,  0.6597,  0.5042,  0.6608,  0.9455};

    for (int i = 0; i < res->dims.size(); i++)
    {
        EXPECT_NEAR(res->data[i], expected[i],0.001);
    }
}


TEST(TensorTest, TensorTanhBackpropCheck)
{
    auto t4 = std::make_shared<synaptic::tensor<float>>(std::vector<int>{2, 2});
    t4->data = { 0.6719,  0.9530, -0.0363,  0.4249,  1.0396, -1.3829, -0.9206, -0.4506,
        -0.8744, -0.3485,  0.9478, -0.3321, -0.6973, -1.0650,  0.1109,  0.1324,
        -0.3065, -0.2386, -0.9518, -0.1970, -0.0411,  0.0774,  0.1147,  0.0699,
        -0.1717,  0.9863, -1.4231,  0.8822,  0.9922,  0.9667};

    synaptic::connections::tanh<float> tanh = synaptic::connections::tanh<float>();
    auto res = tanh.forward(t4);
    //std::cout << *res << std::endl;
    std::vector<float> expected = std::vector<float>{0.6563, 0.4507, 0.9987, 0.8392, 0.3951, 0.2228, 0.4726, 0.8216, 0.5049,
        0.8878, 0.4542, 0.8973, 0.6368, 0.3797, 0.9878, 0.9827, 0.9117, 0.9452,
        0.4515, 0.9622, 0.9983, 0.9940, 0.9870, 0.9951, 0.9711, 0.4288, 0.2074,
        0.4994, 0.4250, 0.4416};

    res->backprop();
    for (int i = 0; i < res->total; i++)
    {
        EXPECT_NEAR(t4->grad[i], expected[i],0.001);
    }
}
