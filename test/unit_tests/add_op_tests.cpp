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
    std::vector<float> expected = {2.0, 7.0};
    for (int i = 0; i < res->total; i++)
    {
        if (res->data[i] != expected[i])
        {
            return false;
        }
    }
    return true;
}

bool test2()
{
    auto t3 = std::make_shared<tensor<int>>(std::vector<int>{5, 3, 2});
    t3->data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30};

    auto res = t3+1;
    std::vector<int> expected = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,31};
    std::cout << *(res) << std::endl; 
    for (int i = 0; i < res->total; i++)
    {
        if (res->data[i] != expected[i])
        {
            return false;
        }
    }
    return true;

}

bool test3()
{
    auto t4 = std::make_shared<tensor<int>>(std::vector<int>{5, 3, 2});
    t4->data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30};

    auto res = 1+t4;
    std::vector<int> expected = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,30, 31};
    for (int i = 0; i < res->total; i++)
    {
        if (res->data[i] != expected[i])
        {
            return false;
        }
    }
    return true;
}

bool test4()
{
    auto t5 = std::make_shared<tensor<float>>(std::vector<int>{5, 3, 2});
    t5->data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30};

    auto res = 1.5f+t5;
    std::vector<float> expected = {2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5, 28.5, 29.5,30.5, 31.5};
    for (int i = 0; i < res->total; i++)
    {
        if (res->data[i] != expected[i])
        {
            return false;
        }
    }
    return true;
}

int main()
{
    if (test1())
    {
        std::cout << "Test 1 passed" << std::endl;
    }
    else
    {
        std::cout << "Test 1 failed" << std::endl;
        return 1;
    }

    if (test2())
    {
        std::cout << "Test 2 passed" << std::endl;
    }
    else
    {
        std::cout << "Test 2 failed" << std::endl;
        return 1;
    }

    if (test3())
    {
        std::cout << "Test 3 passed" << std::endl;
    }
    else
    {
        std::cout << "Test 3 failed" << std::endl;
        return 1;
    }

    if (test4())
    {
        std::cout << "Test 4 passed" << std::endl;
    }
    else
    {
        std::cout << "Test 4 failed" << std::endl;
        return 1;
    }
    return 0;
}