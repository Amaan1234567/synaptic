#include <iostream>
#include <vector>
#include <memory>
#include <cassert>
#include <algorithm>
#include <stdexcept>
#include <limits>
#include <cmath>
#include "op_enum.hpp"
#include "synaptic.hpp"
#include "connections.hpp"
#include "tensor_add.hpp"
#include "tensor_sub.hpp"
#include "tensor_mul.hpp"
#include "tensor_div.hpp"
#include "tensor_matmul.hpp"
#include "tensor_transpose.hpp"
#include "tensor_reshape.hpp"
#include "tensor_pow.hpp"
#include "tensor_exp.hpp"
#include "tensor_log.hpp"


template <typename type>
std::ostream &operator<<(std::ostream &output, const synaptic::tensor<type> &t)
{
    output << "tensor:\nData: \n";
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


template <typename type>
bool synaptic::tensor<type>::dim_check(std::shared_ptr<synaptic::tensor<type>> operand1, std::shared_ptr<synaptic::tensor<type>> operand2)
{
    if (operand1->dims.size() != operand2->dims.size())
    {
        throw std::runtime_error("Number of Dimensions of tensors being added should be the same");
    }
    return true;
}

template <typename type>
bool synaptic::tensor<type>::shape_check(std::shared_ptr<synaptic::tensor<type>> operand1, std::shared_ptr<synaptic::tensor<type>> operand2)
{
    bool shape_flag = true;
    for (int i = 0; i < operand1->dims.size(); ++i)
    {
        if (operand1->dims[i] != operand2->dims[i])
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

template <typename type>
void synaptic::tensor<type>::common_tensor_compatibility_tests(std::shared_ptr<synaptic::tensor<type>> operand1, std::shared_ptr<synaptic::tensor<type>> operand2)
{
    synaptic::tensor<type>::dim_check(operand1, operand2);
    synaptic::tensor<type>::shape_check(operand1, operand2);
}

template <typename type>
std::shared_ptr<synaptic::tensor<type>> synaptic::tensor<type>::add(std::shared_ptr<synaptic::tensor<type>> operand1, std::shared_ptr<synaptic::tensor<type>> operand2)
{
    auto add_impl = std::make_shared<tensor_add<type>>(operand1->device); // Use std::make_shared
    auto output = add_impl->forward(operand1, operand2);
    
    return output;
}

template <typename type>
std::shared_ptr<synaptic::tensor<type>> synaptic::tensor<type>::sub(std::shared_ptr<synaptic::tensor<type>> operand1, std::shared_ptr<synaptic::tensor<type>> operand2)
{
    auto sub_impl = std::make_shared<tensor_sub<type>>(operand1->device); // Use std::make_shared
    auto output = sub_impl->forward(operand1, operand2);
    
    return output;
}

template <typename type>
std::shared_ptr<synaptic::tensor<type>> synaptic::tensor<type>::mul(std::shared_ptr<synaptic::tensor<type>> operand1, std::shared_ptr<synaptic::tensor<type>> operand2)
{
    auto mul_impl = std::make_shared<tensor_mul<type>>(operand1->device); // Use std::make_shared
    auto output = mul_impl->forward(operand1, operand2);
    
    return output;
}

template <typename type>
std::shared_ptr<synaptic::tensor<type>> synaptic::tensor<type>::div(std::shared_ptr<tensor> operand1, std::shared_ptr<tensor> operand2)
{
    auto div_impl = std::make_shared<tensor_div<type>>(operand1->device); // Use std::make_shared
    auto output = div_impl->forward(operand1, operand2);
    
    return output;
}

template <typename type>
std::shared_ptr<synaptic::tensor<type>> synaptic::tensor<type>::pow(std::shared_ptr<tensor> operand1, float pow)
{
    auto div_impl = std::make_shared<tensor_pow<type>>(operand1->device, pow); // Use std::make_shared
    auto output = div_impl->forward(operand1);
    
    return output;
}

template <typename type>
std::shared_ptr<synaptic::tensor<type>> synaptic::tensor<type>::exp(std::shared_ptr<tensor> operand1)
{
    auto div_impl = std::make_shared<tensor_exp<type>>(operand1->device); // Use std::make_shared
    auto output = div_impl->forward(operand1);
    
    return output;
}

template <typename type>
std::shared_ptr<synaptic::tensor<type>> synaptic::tensor<type>::log(std::shared_ptr<tensor> operand1, double base)
{
    auto div_impl = std::make_shared<tensor_log<type>>(operand1->device,base); // Use std::make_shared
    auto output = div_impl->forward(operand1);
    
    return output;
}

template <typename type>
std::shared_ptr<synaptic::tensor<type>> synaptic::tensor<type>::matmul(std::shared_ptr<synaptic::tensor<type>> operand1, std::shared_ptr<synaptic::tensor<type>> operand2)
{
    auto div_impl = std::make_shared<tensor_matmul<type>>(operand1->device); // Use std::make_shared
    auto output = div_impl->forward(operand1, operand2);
    
    return output;
}

template <typename type>
std::shared_ptr<synaptic::tensor<type>> synaptic::tensor<type>::transpose(std::shared_ptr<synaptic::tensor<type>> operand1, int dim0, int dim1)
{
    auto div_impl = std::make_shared<tensor_transpose<type>>(operand1->device, dim0, dim1); // Use std::make_shared
    auto output = div_impl->forward(operand1);
    
    return output;
}

template <typename type>
std::shared_ptr<synaptic::tensor<type>> synaptic::tensor<type>::reshape(std::shared_ptr<synaptic::tensor<type>> operand1, std::vector<int> new_shape)
{
    auto div_impl = std::make_shared<tensor_reshape<type>>(operand1->device, new_shape); // Use std::make_shared
    auto output = div_impl->forward(operand1);
    
    return output;
}

template <typename type>
void synaptic::tensor<type>::recursive_backprop(std::shared_ptr<synaptic::tensor<type>> cur)
{
    
    std::cout << cur->previous_nodes.size() << std::endl;

    // cur->operand_obj_ptr->backward(cur->previous_nodes[0], cur, cur->previous_nodes[1]);
    // std::cout <<  << std::endl;
    if (cur->previous_nodes.size() == 2)
    {
        LOG_DEBUG("tensor class","backward");
        cur->operand_obj_ptr->backward(cur->previous_nodes[0], cur, cur->previous_nodes[1]);
        LOG_DEBUG("tensor class","backward done");

    }
    else if (cur->previous_nodes.size() == 1)
        cur->operand_obj_ptr->backward(cur->previous_nodes[0], cur);
    else
        return;

    LOG_DEBUG("tensor class","recursing");
    if (cur->previous_nodes.size() != 0)
    {
        LOG_DEBUG("tensor class","inside");
        recursive_backprop(cur->previous_nodes[0]);
    }

    if (cur->previous_nodes.size() != 1)
        recursive_backprop(cur->previous_nodes[1]);
}

template <typename type>
void synaptic::tensor<type>::backprop()
{
    this->grad = std::vector<type>(this->total, 1);
    recursive_backprop(std::make_shared<tensor>(*this));
}

template <typename type>
std::shared_ptr<synaptic::tensor<type>> operator+(std::shared_ptr<synaptic::tensor<type>> operand1, std::shared_ptr<synaptic::tensor<type>> operand2)
{
    return synaptic::tensor<type>::add(operand1, operand2);
}

template <typename type>
std::shared_ptr<synaptic::tensor<type>> operator+(std::shared_ptr<synaptic::tensor<type>> operand1, type operand2)
{
    auto tensorised_scalar = std::make_shared<synaptic::tensor<type>>(operand1->dims);
    tensorised_scalar->data = std::vector<type>(tensorised_scalar->total, operand2);
    return synaptic::tensor<type>::add(operand1, tensorised_scalar);
}

template <typename type>
std::shared_ptr<synaptic::tensor<type>> operator+(type operand1, std::shared_ptr<synaptic::tensor<type>> operand2)
{
    auto tensorised_scalar = std::make_shared<synaptic::tensor<type>>(operand2->dims);
    tensorised_scalar->data = std::vector<type>(tensorised_scalar->total, operand1);
    return synaptic::tensor<type>::add(operand2, tensorised_scalar);
}

template <typename type>
std::shared_ptr<synaptic::tensor<type>> operator*(std::shared_ptr<synaptic::tensor<type>> operand1, std::shared_ptr<synaptic::tensor<type>> operand2)
{
    return synaptic::tensor<type>::mul(operand1, operand2);
}

template <typename type>
std::shared_ptr<synaptic::tensor<type>> operator*(std::shared_ptr<synaptic::tensor<type>> operand1, type operand2)
{
    auto tensorised_scalar = std::make_shared<synaptic::tensor<type>>(operand1->dims);
    tensorised_scalar->data = std::vector<type>(tensorised_scalar->total, operand2);
    return synaptic::tensor<type>::mul(operand1, tensorised_scalar);
}

template <typename type>
std::shared_ptr<synaptic::tensor<type>> operator*(type operand1, std::shared_ptr<synaptic::tensor<type>> operand2)
{
    auto tensorised_scalar = std::make_shared<synaptic::tensor<type>>(operand2->dims);
    tensorised_scalar->data = std::vector<type>(tensorised_scalar->total, operand1);
    return synaptic::tensor<type>::mul(tensorised_scalar, operand2);
}

template <typename type>
std::shared_ptr<synaptic::tensor<type>> operator/(std::shared_ptr<synaptic::tensor<type>> operand1, std::shared_ptr<synaptic::tensor<type>> operand2)
{
    return synaptic::tensor<type>::div(operand1, operand2);
}

template <typename type>
std::shared_ptr<synaptic::tensor<type>> operator/(std::shared_ptr<synaptic::tensor<type>> operand1, type operand2)
{
    auto tensorised_scalar = std::make_shared<synaptic::tensor<type>>(operand1->dims);
    tensorised_scalar->data = std::vector<type>(tensorised_scalar->total, operand2);
    return synaptic::tensor<type>::div(operand1, tensorised_scalar);
}

template <typename type>
std::shared_ptr<synaptic::tensor<type>> operator/(type operand1, std::shared_ptr<synaptic::tensor<type>> operand2)
{
    auto tensorised_scalar = std::make_shared<synaptic::tensor<type>>(operand2->dims);
    tensorised_scalar->data = std::vector<type>(tensorised_scalar->total, operand1);
    return synaptic::tensor<type>::div(tensorised_scalar, operand2);
}

template <typename type>
std::shared_ptr<synaptic::tensor<type>> operator-(std::shared_ptr<synaptic::tensor<type>> operand1, std::shared_ptr<synaptic::tensor<type>> operand2)
{
    return synaptic::tensor<type>::sub(operand1, operand2);
}

template <typename type>
std::shared_ptr<synaptic::tensor<type>> operator-(std::shared_ptr<synaptic::tensor<type>> operand1, type operand2)
{
    auto tensorised_scalar = std::make_shared<synaptic::tensor<type>>(operand1->dims);
    tensorised_scalar->data = std::vector<type>(tensorised_scalar->total, operand2);
    return synaptic::tensor<type>::sub(operand1, tensorised_scalar);
}

template <typename type>
std::shared_ptr<synaptic::tensor<type>> operator-(type operand1, std::shared_ptr<synaptic::tensor<type>> operand2)
{
    auto tensorised_scalar = std::make_shared<synaptic::tensor<type>>(operand2->dims);
    tensorised_scalar->data = std::vector<type>(tensorised_scalar->total, operand1);
    return synaptic::tensor<type>::sub(operand2, tensorised_scalar);
}

