#pragma once
#include "../abstracts/basic_op.hpp"
#include "../tensor.hpp"
#include <memory>
#include <functional>
#include <cmath>

namespace synaptic
{

    template <typename type>
    class tensor_log : public basic_op<type>
    {
    public:
        explicit tensor_log(devices dev = devices::none,float base=M_Ef64) : device(dev), base(base) {}

        double base = M_Ef64 ;
        type log_of_base_precalculated = type(1);
        devices device = devices::none;

        // Use std::function instead of function pointer
        using device_specific_forward = std::function<std::shared_ptr<tensor<type>>(std::shared_ptr<tensor<type>>, std::shared_ptr<tensor<type>>)>;
        using device_specific_backward = std::function<void(std::shared_ptr<tensor<type>>, std::shared_ptr<tensor<type>>, std::shared_ptr<tensor<type>>)>;

        device_specific_forward get_impl_selector_forward();
        device_specific_backward get_impl_selector_backward();

        std::shared_ptr<tensor<type>> forward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> operand2 = nullptr);
        void backward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> output, std::shared_ptr<tensor<type>> operand2 = nullptr);

    private:
        std::shared_ptr<tensor<type>> cpu_forward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> operand2);
        void cpu_backward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> output, std::shared_ptr<tensor<type>> operand2 = nullptr);
        std::shared_ptr<tensor<type>> general_forward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> operand2);
        void general_backward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> output, std::shared_ptr<tensor<type>> operand2 = nullptr);
    };
}
#include "../src/operands/tensor_log.tpp"
