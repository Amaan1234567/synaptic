#include "../../include/synaptic.hpp"
#include <gtest/gtest.h>


TEST(TensorTest, TensorSilu)
{
    auto t4 = std::make_shared<synaptic::tensor<float>>(std::vector<int>{5, 4});
    t4->data = { -10.,  -9.,  -8.,  -7.,  -6.,  -5.,  -4.,  -3.,  -2.,  -1.,   0.,   1.,
          2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.};

    synaptic::connections::silu<float> silu = synaptic::connections::silu<float>();
    auto res = silu.forward(t4);
    //std::cout << *res << std::endl;
    std::vector<float> expected = std::vector<float>{ -4.5398e-04, -1.1106e-03, -2.6828e-03, -6.3774e-03, -1.4836e-02,
        -3.3464e-02, -7.1945e-02, -1.4228e-01, -2.3841e-01, -2.6894e-01,
         0.0000e+00,  7.3106e-01,  1.7616e+00,  2.8577e+00,  3.9281e+00,
         4.9665e+00,  5.9852e+00,  6.9936e+00,  7.9973e+00,  8.9989e+00};

    for (int i = 0; i < res->data.size(); i++)
    {
        EXPECT_NEAR(res->data[i], expected[i],0.001);
    }
}


TEST(TensorTest, TensorSiluBackpropCheck)
{
    auto t4 = std::make_shared<synaptic::tensor<float>>(std::vector<int>{5, 4});
    t4->data = {-10.,  -9.,  -8.,  -7.,  -6.,  -5.,  -4.,  -3.,  -2.,  -1.,   0.,   1.,
          2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.};

    synaptic::connections::silu<float> silu = synaptic::connections::silu<float>();
    auto res = silu.forward(t4);
    //std::cout << *res << std::endl;
    std::vector<float> expected = std::vector<float>{-4.0856e-04, -9.8702e-04, -2.3466e-03, -5.4605e-03, -1.2326e-02,
        -2.6547e-02, -5.2665e-02, -8.8104e-02, -9.0784e-02,  7.2329e-02,
         5.0000e-01,  9.2767e-01,  1.0908e+00,  1.0881e+00,  1.0527e+00,
         1.0265e+00,  1.0123e+00,  1.0055e+00,  1.0023e+00,  1.0010e+00};

    res->backprop();
    for (int i = 0; i < res->total; i++)
    {
        EXPECT_NEAR(t4->grad[i], expected[i],0.001);
    }
}
