// tensor.hpp
#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <memory>
#include <iostream>
#include <cassert>
#include <algorithm>
#include "op_enum.hpp"

template <typename type>
class tensor
{
public:
    std::vector<type> data;
    std::vector<std::shared_ptr<tensor<type>>> previous_nodes;
    std::vector<type> grad;
    std::vector<int> dims;
    int total;
    op operation = op::none;

    tensor() : total(1), dims({1}), data({0}), grad({0}) {}
    tensor(const std::vector<int> &shape)
    {
        total = 1;
        for (const auto &ele : shape)
        {
            total *= ele;
        }
        data.resize(total, 0);
        grad.resize(total, 0);
        dims = shape;
    }
    template <typename T>
    friend std::ostream &operator<<(std::ostream &output, const tensor<T> &t);

    static void add_backprop(std::shared_ptr<tensor> &a, std::shared_ptr<tensor> &b, const tensor &output);
    static void sub_backprop(std::shared_ptr<tensor> &a, std::shared_ptr<tensor> &b, const tensor &output);
    static void mul_backprop(std::shared_ptr<tensor> &a, std::shared_ptr<tensor> &b, const tensor &output);
    static void div_backprop(std::shared_ptr<tensor> &a, std::shared_ptr<tensor> &b, const tensor &output);
    static void pow_backprop(std::shared_ptr<tensor> &a, std::shared_ptr<tensor> &b, const tensor &output);
    static void exp_backprop(std::shared_ptr<tensor> &a, const tensor &output);
    static void matmul_backprop(std::shared_ptr<tensor> &a, std::shared_ptr<tensor> &b, const tensor &output);
    static void transpose_backprop(std::shared_ptr<tensor> &a, std::shared_ptr<tensor> &dims_tensor, std::shared_ptr<tensor> &output);
    static void reshape_backprop(std::shared_ptr<tensor> &a, const tensor &output);
    static void matmul_general_impl(std::shared_ptr<tensor<type>> a, std::shared_ptr<tensor<type>> b, std::shared_ptr<tensor<type>> output, std::vector<int> &custom_dims_a, std::vector<int> &custom_dims_b);
    
    static bool dim_check(std::shared_ptr<tensor> a,std::shared_ptr<tensor> b);
    static bool shape_check(std::shared_ptr<tensor> a,std::shared_ptr<tensor> b);
    static void common_tensor_compatibility_tests(std::shared_ptr<tensor> a,std::shared_ptr<tensor> b);

    static std::shared_ptr<tensor> add(std::shared_ptr<tensor> a, std::shared_ptr<tensor> b);
    static std::shared_ptr<tensor> sub(std::shared_ptr<tensor> a, std::shared_ptr<tensor> b);
    static std::shared_ptr<tensor> mul(std::shared_ptr<tensor> a, std::shared_ptr<tensor> b);
    static std::shared_ptr<tensor> div(std::shared_ptr<tensor> a, std::shared_ptr<tensor> b);
    static std::shared_ptr<tensor> pow(std::shared_ptr<tensor> a, float pow);
    static std::shared_ptr<tensor> exp(std::shared_ptr<tensor> a);
    static std::shared_ptr<tensor> matmul(std::shared_ptr<tensor> a, std::shared_ptr<tensor> b);
    static std::shared_ptr<tensor> transpose(std::shared_ptr<tensor> a, int dim0, int dim1);
    static std::shared_ptr<tensor> reshape(std::shared_ptr<tensor> a, std::vector<int> new_shape);

    static void recursive_backprop(std::shared_ptr<tensor> cur);
    void backprop();
};

// Overloaded operators for tensor

template <typename type>
std::shared_ptr<tensor<type>> operator+(std::shared_ptr<tensor<type>> a, std::shared_ptr<tensor<type>> b);

template <typename type>
std::shared_ptr<tensor<type>> operator+(std::shared_ptr<tensor<type>> a, type b);

template <typename type>
std::shared_ptr<tensor<type>> operator+(type a, std::shared_ptr<tensor<type>> b);

template <typename type>
std::shared_ptr<tensor<type>> operator*(std::shared_ptr<tensor<type>> a, std::shared_ptr<tensor<type>> b);

template <typename type>
std::shared_ptr<tensor<type>> operator*(std::shared_ptr<tensor<type>> a, type b);

template <typename type>
std::shared_ptr<tensor<type>> operator*(type a, std::shared_ptr<tensor<type>> b);

template <typename type>
std::shared_ptr<tensor<type>> operator/(std::shared_ptr<tensor<type>> a, std::shared_ptr<tensor<type>> b);

template <typename type>
std::shared_ptr<tensor<type>> operator/(std::shared_ptr<tensor<type>> a, type b);

template <typename type>
std::shared_ptr<tensor<type>> operator/(type a, std::shared_ptr<tensor<type>> b);

template <typename type>
std::shared_ptr<tensor<type>> operator-(std::shared_ptr<tensor<type>> a, std::shared_ptr<tensor<type>> b);

template <typename type>
std::shared_ptr<tensor<type>> operator-(std::shared_ptr<tensor<type>> a, type b);

template <typename type>
std::shared_ptr<tensor<type>> operator-(type a, std::shared_ptr<tensor<type>> b);

#include "tensor.tpp"

#endif // TENSOR_HPP
