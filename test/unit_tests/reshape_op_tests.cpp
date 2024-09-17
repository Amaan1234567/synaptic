#include "../../include/tensor.hpp"
#include <vector>
#include <memory>
#include <gtest/gtest.h>
#include <stdexcept>

TEST(TensorTest, TensorReshape)
{
    auto t4 = std::make_shared<tensor<float>>(std::vector<int>{5, 3, 2});
    t4->data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30};

    auto res = tensor<float>::reshape(t4,{5,6});
    //std::cout << *res << std::endl;
    std::vector<float> expected = {5,6};

    for (int i = 0; i < res->dims.size(); i++)
    {
        EXPECT_EQ(res->dims[i], expected[i]);
    }
}

TEST(TensorTest, TensorReshapeBackpropCheck)
{
    auto t4 = std::make_shared<tensor<float>>(std::vector<int>{5, 3, 2});
    t4->data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30};

    auto res1 = tensor<float>::reshape(t4,{6,5});
    auto res2 = tensor<float>::reshape(res1,{5,6});
    //std::cout << *res2 << std::endl;
    std::vector<float> expected = std::vector<float>(30,1.0);

    res2->backprop();
    for (int i = 0; i < res2->total; i++)
    {
        EXPECT_EQ(t4->grad[i], expected[i]);
    }
}