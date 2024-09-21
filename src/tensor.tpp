#include <iostream>
#include <vector>
#include <memory>
#include <cassert>
#include <algorithm>
#include <stdexcept>
#include <limits>
#include <cmath>
#include "../include/op_enum.hpp"
#include "../include/synaptic.hpp"
#include "../include/connections.hpp"
#include "../include/operands/tensor_add.hpp"
#include "../include/operands/tensor_sub.hpp"
#include "../include/operands/tensor_mul.hpp"
#include "../include/operands/tensor_div.hpp"
#include "../include/operands/tensor_matmul.hpp"
#include "../include/operands/tensor_transpose.hpp"
#include "../include/operands/tensor_reshape.hpp"
#include "../include/operands/tensor_pow.hpp"
#include "../include/operands/tensor_exp.hpp"

template <typename T>
std::ostream &operator<<(std::ostream &output, const synaptic::tensor<T> &t)
{
    output << "Tensor:\nData: \n";
    for (const auto &ele : t.data)
    {
        output << ele << " ";
    }
    output << "\nGrad : \n";
    for (const auto &ele : t.grad)
    {
        output << ele << " ";
    }
    return output;
}

template <typename T>
void synaptic::tensor<T>::add_backprop(std::shared_ptr<synaptic::tensor<T>> &a, std::shared_ptr<synaptic::tensor<T>> &b, const synaptic::tensor<T> &output)
{
    for (int i = 0; i < output.grad.size(); ++i)
    {
        a->grad[i] += output.grad[i];
        b->grad[i] += output.grad[i];
    }
}
template <typename T>
void synaptic::tensor<T>::sub_backprop(std::shared_ptr<synaptic::tensor<T>> &a, std::shared_ptr<synaptic::tensor<T>> &b, const tensor &output)
{
    for (int i = 0; i < output.grad.size(); ++i)
    {
        a->grad[i] += output.grad[i];
        b->grad[i] += -1 * output.grad[i];
    }
}

template <typename T>
void synaptic::tensor<T>::mul_backprop(std::shared_ptr<synaptic::tensor<T>> &a, std::shared_ptr<synaptic::tensor<T>> &b, const tensor &output)
{
    for (int i = 0; i < output.grad.size(); ++i)
    {
        a->grad[i] += b->data[i] * output.grad[i];
        b->grad[i] += a->data[i] * output.grad[i];
    }
}

template <typename T>
void synaptic::tensor<T>::div_backprop(std::shared_ptr<synaptic::tensor<T>> &a, std::shared_ptr<synaptic::tensor<T>> &b, const tensor &output)
{
    for (int i = 0; i < output.grad.size(); ++i)
    {
        a->grad[i] += T(1) / b->data[i] * output.grad[i];
        b->grad[i] += (-(a->data[i]) / std::pow(b->data[i], 2)) * output.grad[i];
    }
}

template <typename T>
void synaptic::tensor<T>::pow_backprop(std::shared_ptr<synaptic::tensor<T>> &a, std::shared_ptr<synaptic::tensor<T>> &pow_tensor, const tensor &output)
{
    float pow = pow_tensor->data[0];
    auto a_raised_to_pow_minus_one = synaptic::tensor<T>::pow(a, pow - 1);

    for (int i = 0; i < output.grad.size(); ++i)
    {
        a->grad[i] += pow * a_raised_to_pow_minus_one->data[i] * output.grad[i];
    }
}

template <typename T>
void synaptic::tensor<T>::exp_backprop(std::shared_ptr<synaptic::tensor<T>> &a, const tensor &output)
{
    for (int i = 0; i < output.grad.size(); ++i)
    {
        a->grad[i] += output.data[i] * output.grad[i];
    }
    // std::cout << "inside exp_backprop" << std::endl;
    // std::cout<< *a << std::endl;
}

template <typename T>
void synaptic::tensor<T>::matmul_backprop(std::shared_ptr<synaptic::tensor<T>> &a, std::shared_ptr<synaptic::tensor<T>> &b, const tensor &output)
{
    auto grad = std::make_shared<tensor>(output.dims);
    grad->data = output.grad;

    auto res1 = synaptic::tensor<T>::matmul(grad, synaptic::tensor<T>::transpose(b, b->dims.size() - 1, b->dims.size() - 2));
    auto res2 = synaptic::tensor<T>::matmul(synaptic::tensor<T>::transpose(a, a->dims.size() - 1, a->dims.size() - 2), grad);

    a->grad = res1->data;
    b->grad = res2->data;
}

template <typename T>
void synaptic::tensor<T>::transpose_backprop(std::shared_ptr<synaptic::tensor<T>> &a, std::shared_ptr<synaptic::tensor<T>> &dims_tensor, std::shared_ptr<synaptic::tensor<T>> &output)
{
    auto grad_tensor = std::make_shared<tensor>(output->dims);
    grad_tensor->data = output->grad;
    auto res = synaptic::tensor<T>::transpose(grad_tensor, int(dims_tensor->data[1]), int(dims_tensor->data[0]));
    for (int i = 0; i < a->total; i++)
    {
        a->grad[i] += res->data[i] * output->grad[i];
    }
}

template <typename T>
void synaptic::tensor<T>::reshape_backprop(std::shared_ptr<synaptic::tensor<T>> &a, const tensor &output)
{
    auto grad_tensor = std::make_shared<tensor>(output.dims);
    grad_tensor->data = output.grad;
    auto res = synaptic::tensor<T>::reshape(grad_tensor, a->dims);
    for (int i = 0; i < a->total; i++)
    {
        a->grad[i] += res->data[i] * output.grad[i];
    }
}

template <typename T>
bool synaptic::tensor<T>::dim_check(std::shared_ptr<synaptic::tensor<T>> a, std::shared_ptr<synaptic::tensor<T>> b)
{
    if (a->dims.size() != b->dims.size())
    {
        throw std::runtime_error("Number of Dimensions of tensors being added should be the same");
    }
    return true;
}

template <typename T>
bool synaptic::tensor<T>::shape_check(std::shared_ptr<synaptic::tensor<T>> a, std::shared_ptr<synaptic::tensor<T>> b)
{
    bool shape_flag = true;
    for (int i = 0; i < a->dims.size(); ++i)
    {
        if (a->dims[i] != b->dims[i])
        {
            shape_flag = false;
            break;
        }
    }
    if (!shape_flag)
    {
        throw std::runtime_error("Shape mismatch during addition");
    }
    return shape_flag;
}

template <typename T>
void synaptic::tensor<T>::common_tensor_compatibility_tests(std::shared_ptr<synaptic::tensor<T>> a, std::shared_ptr<synaptic::tensor<T>> b)
{
    synaptic::tensor<T>::dim_check(a, b);
    synaptic::tensor<T>::shape_check(a, b);
}

template <typename T>
std::shared_ptr<synaptic::tensor<T>> synaptic::tensor<T>::add(std::shared_ptr<synaptic::tensor<T>> a, std::shared_ptr<synaptic::tensor<T>> b)
{
    auto add_impl = std::make_shared<tensor_add<T>>(a->device); // Use std::make_shared
    auto output = add_impl->forward(a, b);
    // std::cout << "add_impl: " << add_impl << std::endl;
    return output;
}

template <typename T>
std::shared_ptr<synaptic::tensor<T>> synaptic::tensor<T>::sub(std::shared_ptr<synaptic::tensor<T>> a, std::shared_ptr<synaptic::tensor<T>> b)
{
    auto sub_impl = std::make_shared<tensor_sub<T>>(a->device); // Use std::make_shared
    auto output = sub_impl->forward(a, b);
    // std::cout << "add_impl: " << add_impl << std::endl;
    return output;
}

template <typename T>
std::shared_ptr<synaptic::tensor<T>> synaptic::tensor<T>::mul(std::shared_ptr<synaptic::tensor<T>> a, std::shared_ptr<synaptic::tensor<T>> b)
{
    auto mul_impl = std::make_shared<tensor_mul<T>>(a->device); // Use std::make_shared
    auto output = mul_impl->forward(a, b);
    // std::cout << "add_impl: " << add_impl << std::endl;
    return output;
}

template <typename T>
std::shared_ptr<synaptic::tensor<T>> synaptic::tensor<T>::div(std::shared_ptr<tensor> a, std::shared_ptr<tensor> b)
{
    auto div_impl = std::make_shared<tensor_div<T>>(a->device); // Use std::make_shared
    auto output = div_impl->forward(a, b);
    // std::cout << "add_impl: " << add_impl << std::endl;
    return output;
}

template <typename T>
std::shared_ptr<synaptic::tensor<T>> synaptic::tensor<T>::pow(std::shared_ptr<tensor> a, float pow)
{
    auto div_impl = std::make_shared<tensor_pow<T>>(a->device, pow); // Use std::make_shared
    auto output = div_impl->forward(a);
    // std::cout << "add_impl: " << add_impl << std::endl;
    return output;
}

template <typename T>
std::shared_ptr<synaptic::tensor<T>> synaptic::tensor<T>::exp(std::shared_ptr<tensor> a)
{
    auto div_impl = std::make_shared<tensor_exp<T>>(a->device); // Use std::make_shared
    auto output = div_impl->forward(a);
    // std::cout << "add_impl: " << add_impl << std::endl;
    return output;
}

template <typename T>
void synaptic::tensor<T>::matmul_general_impl(std::shared_ptr<synaptic::tensor<T>> a, std::shared_ptr<synaptic::tensor<T>> b, std::shared_ptr<synaptic::tensor<T>> output, std::vector<int> &custom_dims_a, std::vector<int> &custom_dims_b)
{
    for (int batch = 0; batch < custom_dims_a[0]; batch++)
    {
        for (int i = 0; i < custom_dims_a[1]; i++)
        {
            for (int j = 0; j < custom_dims_b[2]; j++)
            {
                int output_idx = batch * (custom_dims_a[1] * custom_dims_b[2]) + i * custom_dims_b[2] + j;
                for (int k = 0; k < custom_dims_a[2]; k++)
                {
                    int a_idx = batch * (custom_dims_a[1] * custom_dims_a[2]) + i * custom_dims_a[2] + k;
                    int b_idx = batch * (custom_dims_b[1] * custom_dims_b[2]) + k * custom_dims_b[2] + j;
                    output->data[output_idx] += a->data[a_idx] * b->data[b_idx];
                    // std::cout << output_idx << " " << a_idx << " " << b_idx << std::endl;
                }
            }
            // std::cout << std::endl;
        }
    }
}
template <typename T>
std::shared_ptr<synaptic::tensor<T>> synaptic::tensor<T>::matmul(std::shared_ptr<synaptic::tensor<T>> a, std::shared_ptr<synaptic::tensor<T>> b)
{
    auto div_impl = std::make_shared<tensor_matmul<T>>(a->device); // Use std::make_shared
    auto output = div_impl->forward(a, b);
    // std::cout << "add_impl: " << add_impl << std::endl;
    return output;
}

template <typename T>
std::shared_ptr<synaptic::tensor<T>> synaptic::tensor<T>::transpose(std::shared_ptr<synaptic::tensor<T>> a, int dim0, int dim1)
{
    auto div_impl = std::make_shared<tensor_transpose<T>>(a->device, dim0, dim1); // Use std::make_shared
    auto output = div_impl->forward(a);
    // std::cout << "add_impl: " << add_impl << std::endl;
    return output;
}

template <typename T>
std::shared_ptr<synaptic::tensor<T>> synaptic::tensor<T>::reshape(std::shared_ptr<synaptic::tensor<T>> a, std::vector<int> new_shape)
{
    auto div_impl = std::make_shared<tensor_reshape<T>>(a->device, new_shape); // Use std::make_shared
    auto output = div_impl->forward(a);
    // std::cout << "add_impl: " << add_impl << std::endl;
    return output;
}

template <typename T>
void synaptic::tensor<T>::recursive_backprop(std::shared_ptr<synaptic::tensor<T>> cur)
{

    std::cout << cur->previous_nodes.size() << std::endl;

    // cur->operand_obj_ptr->backward(cur->previous_nodes[0], cur, cur->previous_nodes[1]);
    // std::cout <<  << std::endl;
    if (cur->previous_nodes.size() == 2)
    {
        // std::cout << "backward" << std::endl;
        cur->operand_obj_ptr->backward(cur->previous_nodes[0], cur, cur->previous_nodes[1]);
        // std::cout << "backward done" << std::endl;
    }
    else if (cur->previous_nodes.size() == 1)
        cur->operand_obj_ptr->backward(cur->previous_nodes[0], cur);
    else
        return;

    // std::cout<<"recursing"<<std::endl;
    if (cur->previous_nodes.size() != 0)
    {
        // std::cout<<"inside"<<std::endl;
        recursive_backprop(cur->previous_nodes[0]);
    }

    if (cur->previous_nodes.size() != 1)
        recursive_backprop(cur->previous_nodes[1]);
}

template <typename T>
void synaptic::tensor<T>::backprop()
{
    this->grad = std::vector<T>(this->total, 1);
    recursive_backprop(std::make_shared<tensor>(*this));
}

template <typename T>
std::shared_ptr<synaptic::tensor<T>> operator+(std::shared_ptr<synaptic::tensor<T>> a, std::shared_ptr<synaptic::tensor<T>> b)
{
    return synaptic::tensor<T>::add(a, b);
}

template <typename T>
std::shared_ptr<synaptic::tensor<T>> operator+(std::shared_ptr<synaptic::tensor<T>> a, T b)
{
    auto tensorised_scalar = std::make_shared<synaptic::tensor<T>>(a->dims);
    tensorised_scalar->data = std::vector<T>(tensorised_scalar->total, b);
    return synaptic::tensor<T>::add(a, tensorised_scalar);
}

template <typename T>
std::shared_ptr<synaptic::tensor<T>> operator+(T a, std::shared_ptr<synaptic::tensor<T>> b)
{
    auto tensorised_scalar = std::make_shared<synaptic::tensor<T>>(b->dims);
    tensorised_scalar->data = std::vector<T>(tensorised_scalar->total, a);
    return synaptic::tensor<T>::add(b, tensorised_scalar);
}

template <typename T>
std::shared_ptr<synaptic::tensor<T>> operator*(std::shared_ptr<synaptic::tensor<T>> a, std::shared_ptr<synaptic::tensor<T>> b)
{
    return synaptic::tensor<T>::mul(a, b);
}

template <typename T>
std::shared_ptr<synaptic::tensor<T>> operator*(std::shared_ptr<synaptic::tensor<T>> a, T b)
{
    auto tensorised_scalar = std::make_shared<synaptic::tensor<T>>(a->dims);
    tensorised_scalar->data = std::vector<T>(tensorised_scalar->total, b);
    return synaptic::tensor<T>::mul(a, tensorised_scalar);
}

template <typename T>
std::shared_ptr<synaptic::tensor<T>> operator*(T a, std::shared_ptr<synaptic::tensor<T>> b)
{
    auto tensorised_scalar = std::make_shared<synaptic::tensor<T>>(b->dims);
    tensorised_scalar->data = std::vector<T>(tensorised_scalar->total, a);
    return synaptic::tensor<T>::mul(tensorised_scalar, b);
}

template <typename T>
std::shared_ptr<synaptic::tensor<T>> operator/(std::shared_ptr<synaptic::tensor<T>> a, std::shared_ptr<synaptic::tensor<T>> b)
{
    return synaptic::tensor<T>::div(a, b);
}

template <typename T>
std::shared_ptr<synaptic::tensor<T>> operator/(std::shared_ptr<synaptic::tensor<T>> a, T b)
{
    auto tensorised_scalar = std::make_shared<synaptic::tensor<T>>(a->dims);
    tensorised_scalar->data = std::vector<T>(tensorised_scalar->total, b);
    return synaptic::tensor<T>::div(a, tensorised_scalar);
}

template <typename T>
std::shared_ptr<synaptic::tensor<T>> operator/(T a, std::shared_ptr<synaptic::tensor<T>> b)
{
    auto tensorised_scalar = std::make_shared<synaptic::tensor<T>>(b->dims);
    tensorised_scalar->data = std::vector<T>(tensorised_scalar->total, a);
    return synaptic::tensor<T>::div(tensorised_scalar, b);
}

template <typename T>
std::shared_ptr<synaptic::tensor<T>> operator-(std::shared_ptr<synaptic::tensor<T>> a, std::shared_ptr<synaptic::tensor<T>> b)
{
    return synaptic::tensor<T>::sub(a, b);
}

template <typename T>
std::shared_ptr<synaptic::tensor<T>> operator-(std::shared_ptr<synaptic::tensor<T>> a, T b)
{
    auto tensorised_scalar = std::make_shared<synaptic::tensor<T>>(a->dims);
    tensorised_scalar->data = std::vector<T>(tensorised_scalar->total, b);
    return synaptic::tensor<T>::sub(a, tensorised_scalar);
}

template <typename T>
std::shared_ptr<synaptic::tensor<T>> operator-(T a, std::shared_ptr<synaptic::tensor<T>> b)
{
    auto tensorised_scalar = std::make_shared<synaptic::tensor<T>>(b->dims);
    tensorised_scalar->data = std::vector<T>(tensorised_scalar->total, a);
    return synaptic::tensor<T>::sub(b, tensorised_scalar);
}