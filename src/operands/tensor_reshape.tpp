#include "../../include/operands/tensor_reshape.hpp"
#include "../../include/device_enum.hpp"

namespace synaptic
{

    template <typename type>
    typename tensor_reshape<type>::device_specific_forward tensor_reshape<type>::get_impl_selector_forward()
    {
        if (this->device == devices::cpu)
        {
            return [this](std::shared_ptr<tensor<type>> op1, std::shared_ptr<tensor<type>> op2)
            {
                return this->cpu_forward(op1, op2);
            };
        }
        // reshape checks for other devices (e.g., GPU) as necessary
        return [this](std::shared_ptr<tensor<type>> op1, std::shared_ptr<tensor<type>> op2)
        {
            return this->general_forward(op1, op2);
        };
    }

    template <typename type>
    typename tensor_reshape<type>::device_specific_backward tensor_reshape<type>::get_impl_selector_backward()
    {
        if (this->device == devices::cpu)
        {
            return [this](std::shared_ptr<tensor<type>> op1, std::shared_ptr<tensor<type>> grad, std::shared_ptr<tensor<type>> op2)
            {
                this->cpu_backward(op1, grad, op2);
            };
        }
        // reshape checks for other devices (e.g., GPU) as necessary
        return [this](std::shared_ptr<tensor<type>> op1, std::shared_ptr<tensor<type>> grad, std::shared_ptr<tensor<type>> op2)
        {
            this->general_backward(op1, grad, op2);
        };
    }

    template <typename type>
    std::shared_ptr<tensor<type>> tensor_reshape<type>::forward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> operand2)
    {
        // Get the appropriate implementation based on the current device
        auto impl = get_impl_selector_forward();
        // Call the selected implementation
        auto output = impl(operand1, operand2);
        return output;
    }

    template <typename type>
    void tensor_reshape<type>::backward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> output, std::shared_ptr<tensor<type>> operand2)
    {
        // Get the appropriate implementation based on the current device
        // std::cout << "Performing backward pass for tensor reshapeition" << std::endl;
        auto impl = get_impl_selector_backward();
        // Call the selected implementation
        impl(operand1, output, operand2);
    }

    // The rest of the implementations (cpu_forward, cpu_backward, general_forward, general_backward)
    // remain the same as they were in your original code.

    template <typename type>
    std::shared_ptr<tensor<type>> tensor_reshape<type>::general_forward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> operand2)
    {
        auto output = std::make_shared<synaptic::tensor<type>>(this->new_shape);
        output->data = operand1->data;
        output->operand_obj_ptr = std::make_shared<tensor_reshape<type>>(*this);
        output->previous_nodes.push_back(operand1);

        return output;
    }

    template <typename type>
    std::shared_ptr<tensor<type>> tensor_reshape<type>::cpu_forward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> operand2)
    {
        return tensor_reshape<type>::general_forward(operand1, operand2);
    }

    template <typename type>
    void tensor_reshape<type>::general_backward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> output, std::shared_ptr<tensor<type>> operand2)
    {
        auto grad_tensor = std::make_shared<tensor<type>>(output->dims);
        grad_tensor->data = output->grad;
        auto res = this->forward(grad_tensor);
        for (int i = 0; i < operand1->total; i++)
        {
            operand1->grad[i] += res->data[i] * output->grad[i];
        }
    }

    template <typename type>
    void tensor_reshape<type>::cpu_backward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> output, std::shared_ptr<tensor<type>> operand2)
    {
        return;
    }

} // namespace synaptic