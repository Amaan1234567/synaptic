#pragma once
#include "../abstracts/basic_op.hpp"
#include "../tensor.hpp"
#include <memory>
#include <functional>

namespace synaptic
{
    namespace loss_fn
    {
        template <typename type>
        class cross_entropy_loss : public basic_op<type>
        {
        public:

            cross_entropy_loss(devices dev = devices::none,std::string reduction = "mean") : device(dev), reduction(reduction) {}

            devices device = devices::none;
            std::string reduction = "mean";
            std::shared_ptr<tensor<type>> softmaxed_data_store;
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
}

#include "../src/loss_fn/cross_entropy_loss.tpp"
