#pragma once
#include "../tensor.hpp"
#include "../device_enum.hpp"
#include <memory> // Include for std::shared_ptr
#include <functional> // Include for std::function

namespace synaptic
{   
    template <typename type>
    class tensor;
    template <typename type>
    class basic_op
    {
    public: // Change to public for the interface
        devices device = devices::none;

        // Declare virtual functions to allow overriding
        virtual std::shared_ptr<tensor<type>> forward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> operand2 = nullptr) = 0;
        virtual void backward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> grad_output, std::shared_ptr<tensor<type>> operand2 = nullptr) = 0;

        using device_specific_forward = std::function<std::shared_ptr<tensor<type>>(std::shared_ptr<tensor<type>>, std::shared_ptr<tensor<type>>)>;
        using device_specific_backward = std::function<void(std::shared_ptr<tensor<type>>, std::shared_ptr<tensor<type>>, std::shared_ptr<tensor<type>>)>;

        device_specific_forward get_impl_selector_forward();
        device_specific_backward get_impl_selector_backward();
    };

}

// BASIC_OP_HPP
