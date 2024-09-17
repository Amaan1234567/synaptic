#include "../../include/tensor.hpp"
#include <vector>
#include <memory>
#include <gtest/gtest.h>
#include <stdexcept>

TEST(TensorTest, TensorTranpose)
{
    auto t4 = std::make_shared<tensor<float>>(std::vector<int>{5, 3, 2});
    t4->data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30};

    auto res = tensor<float>::transpose(t4, 2, 1);
    //std::cout << *res << std::endl;
    std::vector<float> expected = {1, 3, 5,
                                   2, 4, 6,
                                   7, 9, 11,
                                   8, 10, 12,
                                   13, 15, 17,
                                   14, 16, 18,
                                   19, 21, 23,
                                   20, 22, 24,
                                   25, 27, 29,
                                   26, 28, 30};

    for (int i = 0; i < res->total; i++)
    {
        EXPECT_NEAR(res->data[i], expected[i], 0.001);
    }
}

TEST(TensorTest, TensorTransposeBackpropCheck)
{
    auto t4 = std::make_shared<tensor<float>>(std::vector<int>{5, 3, 2});
    t4->data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30};

    auto res = tensor<float>::transpose(t4, 1, 0);
    res->backprop();
    std::vector<float> expected = {1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
                                   1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
                                   1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
                                   1.0000, 1.0000, 1.0000};

    for (int i = 0; i < res->total; i++)
    {
        EXPECT_NEAR(t4->grad[i], expected[i], 0.001);
    }
}
