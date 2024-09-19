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
    auto a_raised_to_pow_minus_one = synaptic::tensor<T>::pow(a,pow-1);
    
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
    //std::cout << "inside exp_backprop" << std::endl;
    //std::cout<< *a << std::endl;
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
    auto res = synaptic::tensor<T>::transpose(grad_tensor,int(dims_tensor->data[1]),int(dims_tensor->data[0]));
    for(int i=0;i<a->total;i++)
    {
        a->grad[i] += res->data[i]*output->grad[i];
    }
}

template <typename T>
void synaptic::tensor<T>::reshape_backprop(std::shared_ptr<synaptic::tensor<T>> &a, const tensor &output)
{
    auto grad_tensor = std::make_shared<tensor>(output.dims);
    grad_tensor->data = output.grad;
    auto res = synaptic::tensor<T>::reshape(grad_tensor,a->dims);
    for(int i=0;i<a->total;i++)
    {
        a->grad[i] += res->data[i]*output.grad[i];
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

    synaptic::tensor<T>::common_tensor_compatibility_tests(a, b);
    auto output = std::make_shared<tensor>(a->dims);
    output->operation = op::add;
    output->previous_nodes.push_back(a);
    output->previous_nodes.push_back(b);
    for (int i = 0; i < a->data.size(); ++i)
    {
        output->data[i] = a->data[i] + b->data[i];
    }
    return output;
}

template <typename T>
std::shared_ptr<synaptic::tensor<T>> synaptic::tensor<T>::sub(std::shared_ptr<synaptic::tensor<T>> a, std::shared_ptr<synaptic::tensor<T>> b)
{
    synaptic::tensor<T>::common_tensor_compatibility_tests(a, b);

    auto output = std::make_shared<tensor>(a->dims);
    output->operation = op::sub;
    output->previous_nodes.push_back(a);
    output->previous_nodes.push_back(b);
    for (int i = 0; i < a->data.size(); ++i)
    {
        output->data[i] = a->data[i] - b->data[i];
    }
    return output;
}

template <typename T>
std::shared_ptr<synaptic::tensor<T>> synaptic::tensor<T>::mul(std::shared_ptr<synaptic::tensor<T>> a, std::shared_ptr<synaptic::tensor<T>> b)
{
    synaptic::tensor<T>::common_tensor_compatibility_tests(a, b);

    auto output = std::make_shared<tensor>(a->dims);
    output->operation = op::mul;
    output->previous_nodes.push_back(a);
    output->previous_nodes.push_back(b);
    for (int i = 0; i < a->data.size(); ++i)
    {
        output->data[i] = a->data[i] * b->data[i];
    }
    return output;
}

template <typename T>
std::shared_ptr<synaptic::tensor<T>> synaptic::tensor<T>::div(std::shared_ptr<tensor> a, std::shared_ptr<tensor> b)
{
    synaptic::tensor<T>::common_tensor_compatibility_tests(a, b);

    auto output = std::make_shared<tensor>(a->dims);
    output->operation = op::divi;
    output->previous_nodes.push_back(a);
    output->previous_nodes.push_back(b);
    for (int i = 0; i < a->data.size(); ++i)
    {
        if (b->data[i] == T(0))
            output->data[i] = std::numeric_limits<T>::infinity();
        else
            output->data[i] = a->data[i] / b->data[i];
    }
    // std::cout << *output << std::endl;
    return output;
}

template <typename T>
std::shared_ptr<synaptic::tensor<T>> synaptic::tensor<T>::pow(std::shared_ptr<tensor> a, float pow)
{
    auto pow_tensor = std::make_shared<tensor<float>>(std::vector<int>{1});
    pow_tensor->data[0]=pow;
    auto output = std::make_shared<tensor>(a->dims);
    output->operation = op::pow;
    output->previous_nodes.push_back(a);
    output->previous_nodes.push_back(pow_tensor);
    for (int i = 0; i < a->data.size(); ++i)
    {
        if (a->data[i] == T(0) && pow<=T(-1))
            output->data[i] = std::numeric_limits<T>::infinity();
        else
            output->data[i] = std::pow(a->data[i],pow);
    }
    return output;
}

template <typename T>
std::shared_ptr<synaptic::tensor<T>> synaptic::tensor<T>::exp(std::shared_ptr<tensor> a)
{
    auto pow_tensor = std::make_shared<tensor<float>>(std::vector<int>{1});
    auto output = std::make_shared<tensor>(a->dims);
    output->operation = op::exp;
    output->previous_nodes.push_back(a);
    for (int i = 0; i < a->data.size(); ++i)
    {
        output->data[i] = std::exp(a->data[i]);
    }
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
                    std::cout << output_idx << " " << a_idx << " " << b_idx << std::endl;
                }
            }
            std::cout << std::endl;
        }
    }
}
template <typename T>
std::shared_ptr<synaptic::tensor<T>> synaptic::tensor<T>::matmul(std::shared_ptr<synaptic::tensor<T>> a, std::shared_ptr<synaptic::tensor<T>> b)
{
    synaptic::tensor<T>::dim_check(a, b);
    bool matmul_flag = true;
    auto last_dim = a->dims.size() - 1;
    if (a->dims[last_dim] != b->dims[last_dim - 1])
        matmul_flag = false;

    if (!matmul_flag)
    {
        std::runtime_error("Tensors not compatible for matrix multiplication please check shapes");
    }

    std::vector<int> custom_dims_a(3, 0);
    std::vector<int> custom_dims_b(3, 0);
    custom_dims_a[2] = a->dims[a->dims.size() - 1];
    custom_dims_a[1] = a->dims[a->dims.size() - 2];
    custom_dims_b[2] = b->dims[b->dims.size() - 1];
    custom_dims_b[1] = b->dims[b->dims.size() - 2];

    int total1 = 1;
    int total2 = 1;
    for (int i = 0; i < a->dims.size() - 2; i++)
    {
        total1 *= a->dims[i];
        total2 *= b->dims[i];
    }
    if (total1 == total2)
    {
        std::runtime_error("Tensors are do not have same shape");
    }

    custom_dims_a[0] = total1;
    custom_dims_b[0] = total2;

    std::vector<int> output_dims(a->dims.begin(), a->dims.end());
    output_dims[output_dims.size() - 1] = b->dims[b->dims.size() - 1];
    std::cout << "output dims: ";
    for (auto dims : output_dims)
    {
        std::cout << dims << " ";
    }
    std::cout << std::endl;
    auto output = std::make_shared<tensor>(output_dims);
    output->operation = op::matmul;
    output->previous_nodes.push_back(a);
    output->previous_nodes.push_back(b);

    matmul_general_impl(a, b, output, custom_dims_a, custom_dims_b);
    return output;
}

template <typename T>
std::shared_ptr<synaptic::tensor<T>> synaptic::tensor<T>::transpose(std::shared_ptr<synaptic::tensor<T>> a, int dim0, int dim1)
{
    if (dim0 >= 0 && dim0 < a->dims.size() && dim1 >= 0 && dim1 < a->dims.size() && dim0 != dim1)
    {
        std::runtime_error("Please provide proper dimension indices, and make sure both the indices are different :)");
    }

    std::vector<int> output_dims = a->dims;
    std::swap(output_dims[dim0], output_dims[dim1]);

    auto output = std::make_shared<tensor>(output_dims);
    auto dims_tensor = std::make_shared<synaptic::tensor<T>>(std::vector<int>{2});
    dims_tensor->data[0]=dim0;
    dims_tensor->data[1]=dim1;
    output->previous_nodes.push_back(a);
    output->previous_nodes.push_back(dims_tensor);
    output->operation = op::transpose;
    
    // Calculate original strides
    std::vector<int> strides(a->dims.size(), 1);
    for (int i = a->dims.size() - 2; i >= 0; i--)
    {
        strides[i] = strides[i + 1] * a->dims[i + 1];
    }

    // Transposed strides
    std::swap(strides[dim0], strides[dim1]);

    // Perform the transpose by iterating through the output tensor
    std::vector<int> counts(a->dims.size(), 0); // Track the current multi-dimensional index
    int total_elements = output->total;

    for (int count = 0; count < total_elements; ++count)
    {
        int original_idx = 0;
        for (int i = 0; i < strides.size(); ++i)
        {
            original_idx += counts[i] * strides[i];
        }
        output->data[count] = a->data[original_idx];

        // Increment the multi-dimensional index
        for (int i = strides.size() - 1; i >= 0; --i)
        {
            counts[i]++;
            if (counts[i] < output->dims[i])
            {
                break;
            }
            counts[i] = 0;
        }
    }

    return output;
}

template <typename T>
std::shared_ptr<synaptic::tensor<T>> synaptic::tensor<T>::reshape(std::shared_ptr<synaptic::tensor<T>> a, std::vector<int> new_shape)
{
    auto output = std::make_shared<synaptic::tensor<T>>(new_shape);
    output->data = a->data;
    output->operation = op::reshape;
    output->previous_nodes.push_back(a);

    return output;
}

template <typename T>
void synaptic::tensor<T>::recursive_backprop(std::shared_ptr<synaptic::tensor<T>> cur)
{
    
    if (cur->operation == op::add)
    {
        add_backprop(cur->previous_nodes[0], cur->previous_nodes[1], *cur);
    }
    else if (cur->operation == op::sub)
    {
        sub_backprop(cur->previous_nodes[0], cur->previous_nodes[1], *cur);
    }
    else if (cur->operation == op::mul)
    {
        mul_backprop(cur->previous_nodes[0], cur->previous_nodes[1], *cur);
    }
    else if (cur->operation == op::divi)
    {
        div_backprop(cur->previous_nodes[0], cur->previous_nodes[1], *cur);
    }
    else if (cur->operation == op::pow)
    {
        pow_backprop(cur->previous_nodes[0], cur->previous_nodes[1], *cur);
    }
    else if (cur->operation == op::exp)
    {
        //std::cout<<"cur operation: "<<"exp"<<std::endl;
        exp_backprop(cur->previous_nodes[0], *cur);
    }
    else if (cur->operation == op::matmul)
    {
        matmul_backprop(cur->previous_nodes[0], cur->previous_nodes[1], *cur);
    }
    else if (cur->operation == op::transpose)
    {
        transpose_backprop(cur->previous_nodes[0], cur->previous_nodes[1], cur);
    }
    else if (cur->operation == op::reshape)
    {
        reshape_backprop(cur->previous_nodes[0],*cur);
    }
    else if (cur->operation == op::relu)
    {
        connections::relu<T>::backward(cur->previous_nodes[0],cur);
    }
    //std::cout<<"recursing"<<std::endl;
    if (cur->previous_nodes.size()!=0 && cur->previous_nodes[0]->operation != op::none)
        recursive_backprop(cur->previous_nodes[0]);

    if (cur->previous_nodes.size()!=1 && cur->previous_nodes[1]->operation != op::none)
        recursive_backprop(cur->previous_nodes[1]);
}

template <typename T>
void synaptic::tensor<T>::backprop()
{
    this->grad = std::vector<T>(this->total, 1);
    recursive_backprop(std::make_shared<tensor>(*this));
}

;

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