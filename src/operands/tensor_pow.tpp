#include "../../include/operands/tensor_pow.hpp"
#include "../../include/device_enum.hpp"

namespace synaptic
{

    template <typename type>
    typename tensor_pow<type>::device_specific_forward tensor_pow<type>::get_impl_selector_forward()
    {
        if (this->device == devices::cpu)
        {
            return [this](std::shared_ptr<tensor<type>> op1, std::shared_ptr<tensor<type>> op2)
            {
                return this->cpu_forward(op1, op2);
            };
        }
        // pow checks for other devices (e.g., GPU) as necessary
        return [this](std::shared_ptr<tensor<type>> op1, std::shared_ptr<tensor<type>> op2)
        {
            return this->general_forward(op1, op2);
        };
    }

    template <typename type>
    typename tensor_pow<type>::device_specific_backward tensor_pow<type>::get_impl_selector_backward()
    {
        if (this->device == devices::cpu)
        {
            return [this](std::shared_ptr<tensor<type>> op1, std::shared_ptr<tensor<type>> grad, std::shared_ptr<tensor<type>> op2)
            {
                this->cpu_backward(op1, grad, op2);
            };
        }
        // pow checks for other devices (e.g., GPU) as necessary
        return [this](std::shared_ptr<tensor<type>> op1, std::shared_ptr<tensor<type>> grad, std::shared_ptr<tensor<type>> op2)
        {
            this->general_backward(op1, grad, op2);
        };
    }

    template <typename type>
    std::shared_ptr<tensor<type>> tensor_pow<type>::forward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> operand2)
    {
        // Get the appropriate implementation based on the current device
        auto impl = get_impl_selector_forward();
        // Call the selected implementation
        auto output = impl(operand1, operand2);
        return output;
    }

    template <typename type>
    void tensor_pow<type>::backward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> output, std::shared_ptr<tensor<type>> operand2)
    {
        // Get the appropriate implementation based on the current device
        // std::cout << "Performing backward pass for tensor powition" << std::endl;
        auto impl = get_impl_selector_backward();
        // Call the selected implementation
        impl(operand1, output, operand2);
    }

    // The rest of the implementations (cpu_forward, cpu_backward, general_forward, general_backward)
    // remain the same as they were in your original code.

    template <typename type>
    std::shared_ptr<tensor<type>> tensor_pow<type>::general_forward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> operand2)
    {
        auto pow_tensor = std::make_shared<tensor<float>>(std::vector<int>{1});
        pow_tensor->data[0] = pow;
        auto output = std::make_shared<tensor<type>>(operand1->dims);
        output->operand_obj_ptr = std::make_shared<tensor_pow<type>>(*this);
        output->previous_nodes.push_back(operand1);
        for (int i = 0; i < operand1->data.size(); ++i)
        {
            if (operand1->data[i] == type(0) && this->pow <= type(-1))
                output->data[i] = std::numeric_limits<type>::infinity();
            else
                output->data[i] = std::pow(operand1->data[i], this->pow);
        }
        return output;
    }

    template <typename type>
    std::shared_ptr<tensor<type>> tensor_pow<type>::cpu_forward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> operand2)
    {
        return tensor_pow<type>::general_forward(operand1, operand2);
    }

    template <typename type>
    void tensor_pow<type>::general_backward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> output, std::shared_ptr<tensor<type>> operand2)
    {
        auto pow_impl = std::make_shared<tensor_pow<type>>(this->device, this->pow - 1);
        auto operand1_raised_to_pow_minus_one = pow_impl->forward(operand1);

        for (int i = 0; i < output->grad.size(); ++i)
        {
            operand1->grad[i] += this->pow * operand1_raised_to_pow_minus_one->data[i] * output->grad[i];
        }
    }

    template <typename type>
    void tensor_pow<type>::cpu_backward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> output, std::shared_ptr<tensor<type>> operand2)
    {
        return;
    }

} // namespace synaptic