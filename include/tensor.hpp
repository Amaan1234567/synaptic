// tensor.hpp
#pragma once
#include <vector>
#include <memory>
#include <iostream>
#include <cassert>
#include <algorithm>
#include "op_enum.hpp"
#include "device_enum.hpp"
#include "abstracts/basic_op.hpp"


namespace synaptic
{
    template <typename type>
    class tensor
    {
    public:
        std::vector<type> data;
        std::vector<std::shared_ptr<tensor<type>>> previous_nodes;
        std::vector<type> grad;
        std::vector<int> dims;
        int total;
        devices device = devices::none;
        std::shared_ptr<basic_op<type>> operand_obj_ptr=nullptr;
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
        

        /* // Move assignment operator
        tensor& operator=(tensor&& other) noexcept
        {
            if (this != &other)
            {
                data = std::move(other.data);
                previous_nodes = std::move(other.previous_nodes);
                grad = std::move(other.grad);
                dims = std::move(other.dims);
                total = other.total;
                operand_obj_ptr = std::move(other.operand_obj_ptr);
            }
            return *this;
        } */

        

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

        static bool dim_check(std::shared_ptr<tensor> a, std::shared_ptr<tensor> b);
        static bool shape_check(std::shared_ptr<tensor> a, std::shared_ptr<tensor> b);
        static void common_tensor_compatibility_tests(std::shared_ptr<tensor> a, std::shared_ptr<tensor> b);

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

}

// Overloaded operators for tensor
template <typename type>
std::ostream &operator<<(std::ostream &output, const synaptic::tensor<type> &t);

template <typename type>
std::shared_ptr<synaptic::tensor<type>> operator+(std::shared_ptr<synaptic::tensor<type>> a, std::shared_ptr<synaptic::tensor<type>> b);

template <typename type>
std::shared_ptr<synaptic::tensor<type>> operator+(std::shared_ptr<synaptic::tensor<type>> a, type b);

template <typename type>
std::shared_ptr<synaptic::tensor<type>> operator+(type a, std::shared_ptr<synaptic::tensor<type>> b);

template <typename type>
std::shared_ptr<synaptic::tensor<type>> operator*(std::shared_ptr<synaptic::tensor<type>> a, std::shared_ptr<synaptic::tensor<type>> b);

template <typename type>
std::shared_ptr<synaptic::tensor<type>> operator*(std::shared_ptr<synaptic::tensor<type>> a, type b);

template <typename type>
std::shared_ptr<synaptic::tensor<type>> operator*(type a, std::shared_ptr<synaptic::tensor<type>> b);

template <typename type>
std::shared_ptr<synaptic::tensor<type>> operator/(std::shared_ptr<synaptic::tensor<type>> a, std::shared_ptr<synaptic::tensor<type>> b);

template <typename type>
std::shared_ptr<synaptic::tensor<type>> operator/(std::shared_ptr<synaptic::tensor<type>> a, type b);

template <typename type>
std::shared_ptr<synaptic::tensor<type>> operator/(type a, std::shared_ptr<synaptic::tensor<type>> b);

template <typename type>
std::shared_ptr<synaptic::tensor<type>> operator-(std::shared_ptr<synaptic::tensor<type>> a, std::shared_ptr<synaptic::tensor<type>> b);

template <typename type>
std::shared_ptr<synaptic::tensor<type>> operator-(std::shared_ptr<synaptic::tensor<type>> a, type b);

template <typename type>
std::shared_ptr<synaptic::tensor<type>> operator-(type a, std::shared_ptr<synaptic::tensor<type>> b);

#include "../src/tensor.tpp"
