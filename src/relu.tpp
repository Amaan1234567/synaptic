#include <memory>
#include "synaptic.hpp"
#include <iostream>

template <typename type>
std::shared_ptr<synaptic::tensor<type>> synaptic::connections::relu<type>::forward(std::shared_ptr<synaptic::tensor<type>> &input_tensor)
{
    auto output_tensor = std::make_shared<tensor<type>>(input_tensor->dims);
    output_tensor->previous_nodes.push_back(input_tensor);
    output_tensor->operation = op::relu;
    for (int i = 0; i < input_tensor->total; i++)
    {
        if (input_tensor->data[i] > type(0))
            output_tensor->data[i] = input_tensor->data[i] * synaptic::connections::relu<type>::non_linearity_multiplier;
        else
            output_tensor->data[i] = synaptic::connections::relu<type>::below_thres_value; // Ensure uninitialized data is handled
    }
    return output_tensor;
}

template <typename type>
void synaptic::connections::relu<type>::backward(std::shared_ptr<synaptic::tensor<type>> &input_tensor, std::shared_ptr<synaptic::tensor<type>> &output_tensor)
{
    for (int i = 0; i < output_tensor->total; i++)
    {
        type grad_multiplier = (input_tensor->data[i] > type(0))
            ? synaptic::connections::relu<type>::non_linearity_multiplier
            : synaptic::connections::relu<type>::below_thres_value;

        input_tensor->grad[i] += grad_multiplier * output_tensor->grad[i];
        std::cout<< *input_tensor <<std::endl;
    }
}
