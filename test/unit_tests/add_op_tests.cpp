#include "../../include/tensor.hpp"
#include <vector>
#include <memory>
#include <iostream>


bool test1()
{
    auto t1 = std::make_shared<tensor<float>>(std::vector<int>{2});
    auto t2 = std::make_shared<tensor<float>>(std::vector<int>{2});
    t1->data = {1.0, 3.0};
    t2->data = {1.0, 4.0};

    auto res = t1+t2 ;
    std::cout << *(res) << std::endl;
    return true;
}

int main()
{
    return !test1();
}