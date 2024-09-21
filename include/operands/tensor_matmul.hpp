#pragma once
#include "../abstracts/basic_op.hpp"
#include "../tensor.hpp"
#include <memory>
#include <functional>

namespace synaptic
{

    template <typename type>
    class tensor_matmul : public basic_op<type>
    {
    public:
        explicit tensor_matmul(devices dev = devices::none) : device(dev) {}

        devices device = devices::none;

        // Use std::function instead of function pointer
        using device_specific_forward = std::function<std::shared_ptr<tensor<type>>(std::shared_ptr<tensor<type>>, std::shared_ptr<tensor<type>>)>;
        using device_specific_backward = std::function<void(std::shared_ptr<tensor<type>>, std::shared_ptr<tensor<type>>, std::shared_ptr<tensor<type>>)>;

        device_specific_forward get_impl_selector_forward();
        device_specific_backward get_impl_selector_backward();

        std::shared_ptr<tensor<type>> forward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> operand2 = nullptr);
        void backward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> output, std::shared_ptr<tensor<type>> operand2 = nullptr);

    private:
        bool dim_check(std::shared_ptr<synaptic::tensor<type>> operand1, std::shared_ptr<synaptic::tensor<type>> operand2);
        void matmul_general_impl(std::shared_ptr<synaptic::tensor<type>> operand1, std::shared_ptr<synaptic::tensor<type>> operand2, std::shared_ptr<synaptic::tensor<type>> output, std::vector<int> &custom_dims_a, std::vector<int> &custom_dims_b);
        std::shared_ptr<tensor<type>> cpu_forward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> operand2);
        void cpu_backward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> output, std::shared_ptr<tensor<type>> operand2 = nullptr);
        std::shared_ptr<tensor<type>> general_forward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> operand2);
        void general_backward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> output, std::shared_ptr<tensor<type>> operand2 = nullptr);
    };
}
#include "../src/operands/tensor_matmul.tpp"
