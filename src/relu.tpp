#include <memory>
#include "synaptic.hpp"

using namespace synaptic;

template <typename type>
std::shared_ptr<synaptic::tensor<type>> connections::relu<type>::forward(const std::shared_ptr<synaptic::tensor<type>> &a)
{
    auto output = std::make_shared<tensor<type>>(a->dims);

    for (int i = 0; i < a->total; i++)
    {
        if(a->data[i]> connections::relu<type>::below_thres_value)
        output->data[i] = a->data[i]*connections::relu<type>::non_linearity_multiplier;
    }
    return output;
}