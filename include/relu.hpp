#ifndef RELU_HPP
#define RELU_HPP

#include <memory>

namespace synaptic
{
    namespace connections
    {
        template <typename type>
        class relu
        {
        public:
            relu() = default;
            /* relu(type val) : non_linearity_multiplier(val), below_thres_value(type(0)) {}
            relu(type val1, type val2) : non_linearity_multiplier(val1), below_thres_value(val2) {} */

            static constexpr type non_linearity_multiplier = type(1);
            static constexpr type below_thres_value = type(0);

            std::shared_ptr<synaptic::tensor<type>> forward(std::shared_ptr<synaptic::tensor<type>> &input_tensor);
            static void backward(std::shared_ptr<synaptic::tensor<type>> &input_tensor, std::shared_ptr<synaptic::tensor<type>> &output_tensor);
        };
    } // namespace connections
} // namespace synaptic

#include "../src/relu.tpp"
#endif
