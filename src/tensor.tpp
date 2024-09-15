#include <iostream>
#include <vector>
#include <memory>
#include <cassert>
#include <algorithm>
#include <stdexcept>
#include "../include/op_enum.hpp"
//#include "../include/tensor.hpp"


template <typename T>
std::ostream &operator<<(std::ostream &output, const tensor<T> &t)
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
void tensor<T>::add_backprop(std::shared_ptr<tensor<T>> &a, std::shared_ptr<tensor<T>> &b, const tensor<T> &output)
{
    for (int i = 0; i < output.grad.size(); ++i)
    {
        a->grad[i] += output.grad[i];
        b->grad[i] += output.grad[i];
    }
}
template <typename T>
void tensor<T>::sub_backprop(std::shared_ptr<tensor<T>> &a, std::shared_ptr<tensor<T>> &b, const tensor &output)
{
    for (int i = 0; i < output.grad.size(); ++i)
    {
        a->grad[i] += output.grad[i];
        b->grad[i] += -1 * output.grad[i];
    }
}
template <typename T>
void tensor<T>::mul_backprop(std::shared_ptr<tensor<T>> &a, std::shared_ptr<tensor<T>> &b, const tensor &output)
{
    for (int i = 0; i < output.grad.size(); ++i)
    {
        a->grad[i] += b->data[i] * output.grad[i];
        b->grad[i] += a->data[i] * output.grad[i];
    }
}
template <typename T>
void tensor<T>::matmul_backprop(std::shared_ptr<tensor<T>> &a, std::shared_ptr<tensor<T>> &b, const tensor &output)
{
    auto grad = std::make_shared<tensor>(output.dims);
    grad->data = output.grad;

    auto res1 = tensor<T>::matmul(grad, tensor<T>::transpose(b, b->dims.size() - 1, b->dims.size() - 2));
    auto res2 = tensor<T>::matmul(tensor<T>::transpose(a, a->dims.size() - 1, a->dims.size() - 2), grad);

    a->grad = res1->data;
    b->grad = res2->data;
}
template <typename T>
std::shared_ptr<tensor<T>> tensor<T>::add(std::shared_ptr<tensor<T>> a, std::shared_ptr<tensor<T>> b)
{
    
    if (a->dims.size() != b->dims.size()) {
            throw std::runtime_error("Number of Dimensions of tensors being added should be the same");
    }
    
    bool shape_flag = true;
    for (int i = 0; i < a->dims.size(); ++i)
    {
        if (a->dims[i] != b->dims[i])
        {
            shape_flag = false;
            break;
        }
    }
    if (!shape_flag) {
            throw std::runtime_error("Shape mismatch during addition");
    }

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
std::shared_ptr<tensor<T>> tensor<T>::mul(std::shared_ptr<tensor<T>> a, std::shared_ptr<tensor<T>> b)
{
    assert(a->dims.size() == b->dims.size() && "Number of Dimensions of tensors being added should be the same");
    bool shape_flag = true;
    for (int i = 0; i < a->dims.size(); ++i)
    {
        if (a->dims[i] != b->dims[i])
        {
            shape_flag = false;
            break;
        }
    }
    assert(shape_flag && "Shape of Tensors not same");

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
std::shared_ptr<tensor<T>> tensor<T>::sub(std::shared_ptr<tensor<T>> a, std::shared_ptr<tensor<T>> b)
{
    assert(a->dims.size() == b->dims.size() && "Number of Dimensions of tensors being added should be the same");
    bool shape_flag = true;
    for (int i = 0; i < a->dims.size(); ++i)
    {
        if (a->dims[i] != b->dims[i])
        {
            shape_flag = false;
            break;
        }
    }
    assert(shape_flag && "Shape of Tensors not same");

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
void tensor<T>::matmul_general_impl(std::shared_ptr<tensor<T>> a, std::shared_ptr<tensor<T>> b, std::shared_ptr<tensor<T>> output, std::vector<int> &custom_dims_a, std::vector<int> &custom_dims_b)
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
std::shared_ptr<tensor<T>> tensor<T>::matmul(std::shared_ptr<tensor<T>> a, std::shared_ptr<tensor<T>> b)
{
    assert(a->dims.size() == b->dims.size() && "Number of Dimensions of tensors being added should be the same");
    bool matmul_flag = true;
    auto last_dim = a->dims.size() - 1;
    if (a->dims[last_dim] != b->dims[last_dim - 1])
        matmul_flag = false;

    assert(matmul_flag && "Tensors not compatible for matrix multiplication please check shapes");

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
    assert(total1 == total2 && "Tensors are do not have same shape");

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
std::shared_ptr<tensor<T>> tensor<T>::transpose(std::shared_ptr<tensor<T>> a, int dim0, int dim1)
{
    assert(dim0 >= 0 && dim0 < a->dims.size() && dim1 >= 0 && dim1 < a->dims.size() && dim0 != dim1 && "Please provide proper dimension indices, and make sure both the indices are different :)");

    std::vector<int> output_dims = a->dims;
    std::swap(output_dims[dim0], output_dims[dim1]);

    auto output = std::make_shared<tensor>(output_dims);

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
        std::cout << "counts: ";
        for (auto ele : counts)
            std::cout << ele << " ";
        std::cout << std::endl;
        for (int i = 0; i < strides.size(); ++i)
        {
            original_idx += counts[i] * strides[i];
        }
        std::cout << "idx: " << original_idx << std::endl;
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
void tensor<T>::recursive_backprop(std::shared_ptr<tensor<T>> cur)
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
    else if (cur->operation == op::matmul)
    {
        matmul_backprop(cur->previous_nodes[0], cur->previous_nodes[1], *cur);
    }

    if (cur->previous_nodes[0]->operation != op::none)
        recursive_backprop(cur->previous_nodes[0]);

    if (cur->previous_nodes[1]->operation != op::none)
        recursive_backprop(cur->previous_nodes[1]);
}
template <typename T>
void tensor<T>::backprop()
{
    this->grad = std::vector<T>(this->total, 1);
    recursive_backprop(std::make_shared<tensor>(*this));
}

;

template <typename T>
std::shared_ptr<tensor<T>> operator+(std::shared_ptr<tensor<T>> a, std::shared_ptr<tensor<T>> b)
{
    return tensor<T>::add(a, b);
}

template <typename T>
std::shared_ptr<tensor<T>> operator+(std::shared_ptr<tensor<T>> a, T b)
{
    auto tensorised_scalar = std::make_shared<tensor<T>>(a->dims);
    tensorised_scalar->data = std::vector<T>(tensorised_scalar->total, b);
    return tensor<T>::add(a, tensorised_scalar);
}

template <typename T>
std::shared_ptr<tensor<T>> operator+(T a, std::shared_ptr<tensor<T>> b)
{
    auto tensorised_scalar = std::make_shared<tensor<T>>(b->dims);
    tensorised_scalar->data = std::vector<T>(tensorised_scalar->total, a);
    return tensor<T>::add(b, tensorised_scalar);
}

template <typename T>
std::shared_ptr<tensor<T>> operator*(std::shared_ptr<tensor<T>> a, std::shared_ptr<tensor<T>> b)
{
    return tensor<T>::mul(a, b);
}

template <typename T>
std::shared_ptr<tensor<T>> operator*(std::shared_ptr<tensor<T>> a, T b)
{
    auto tensorised_scalar = std::make_shared<tensor<T>>(a->dims);
    tensorised_scalar->data = std::vector<T>(tensorised_scalar->total, b);
    return tensor<T>::mul(a, tensorised_scalar);
}

template <typename T>
std::shared_ptr<tensor<T>> operator*(T a, std::shared_ptr<tensor<T>> b)
{
    auto tensorised_scalar = std::make_shared<tensor<T>>(b->dims);
    tensorised_scalar->data = std::vector<T>(tensorised_scalar->total, a);
    return tensor<T>::mul(b, tensorised_scalar);
}

template <typename T>
std::shared_ptr<tensor<T>> operator-(std::shared_ptr<tensor<T>> a, std::shared_ptr<tensor<T>> b)
{
    return tensor<T>::sub(a, b);
}

template <typename T>
std::shared_ptr<tensor<T>> operator-(std::shared_ptr<tensor<T>> a, T b)
{
    auto tensorised_scalar = std::make_shared<tensor<T>>(a->dims);
    tensorised_scalar->data = std::vector<T>(tensorised_scalar->total, b);
    return tensor<T>::sub(a, tensorised_scalar);
}

template <typename T>
std::shared_ptr<tensor<T>> operator-(T a, std::shared_ptr<tensor<T>> b)
{
    auto tensorised_scalar = std::make_shared<tensor<T>>(b->dims);
    tensorised_scalar->data = std::vector<T>(tensorised_scalar->total, a);
    return tensor<T>::sub(b, tensorised_scalar);
}