#include "../../include/operands/tensor_sub.hpp"
#include "../../include/device_enum.hpp"

namespace synaptic
{

    template <typename type>
    typename tensor_sub<type>::device_specific_forward tensor_sub<type>::get_impl_selector_forward()
    {
        if (this->device == devices::cpu)
        {
            return [this](std::shared_ptr<tensor<type>> op1, std::shared_ptr<tensor<type>> op2)
            {
                return this->cpu_forward(op1, op2);
            };
        }
        // sub checks for other devices (e.g., GPU) as necessary
        return [this](std::shared_ptr<tensor<type>> op1, std::shared_ptr<tensor<type>> op2)
        {
            return this->general_forward(op1, op2);
        };
    }

    template <typename type>
    typename tensor_sub<type>::device_specific_backward tensor_sub<type>::get_impl_selector_backward()
    {
        if (this->device == devices::cpu)
        {
            return [this](std::shared_ptr<tensor<type>> op1, std::shared_ptr<tensor<type>> grad, std::shared_ptr<tensor<type>> op2)
            {
                this->cpu_backward(op1, grad, op2);
            };
        }
        // sub checks for other devices (e.g., GPU) as necessary
        return [this](std::shared_ptr<tensor<type>> op1, std::shared_ptr<tensor<type>> grad, std::shared_ptr<tensor<type>> op2)
        {
            this->general_backward(op1, grad, op2);
        };
    }

    template <typename type>
    std::shared_ptr<tensor<type>> tensor_sub<type>::forward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> operand2)
    {
        if (operand2 == nullptr)
        {
            return operand1; // Return the first operand if the second is not provided
        }
        // Get the appropriate implementation based on the current device
        auto impl = get_impl_selector_forward();
        // Call the selected implementation
        auto output = impl(operand1, operand2);
        return output;
    }

    template <typename type>
    void tensor_sub<type>::backward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> output, std::shared_ptr<tensor<type>> operand2)
    {
        // Get the appropriate implementation based on the current device
        // std::cout << "Performing backward pass for tensor subition" << std::endl;
        auto impl = get_impl_selector_backward();
        // Call the selected implementation
        impl(operand1, output, operand2);
    }

    // The rest of the implementations (cpu_forward, cpu_backward, general_forward, general_backward)
    // remain the same as they were in your original code.

    template <typename type>
    std::shared_ptr<tensor<type>> tensor_sub<type>::general_forward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> operand2)
    {
        // Perform compatibility tests
        synaptic::tensor<type>::common_tensor_compatibility_tests(operand1, operand2);
        auto output = std::make_shared<tensor<type>>(operand1->dims);
        output->previous_nodes.push_back(operand1);
        output->previous_nodes.push_back(operand2);
        output->operand_obj_ptr = std::make_shared<tensor_sub<type>>(*this);
        // output->operand_obj_ptr = this;

        // std::cout << "subress of new pointer created: "<< output->operand_obj_ptr << std::endl;
        //  Assuming both operands have the same size
        for (size_t i = 0; i < operand1->data.size(); ++i)
        {
            output->data[i] = operand1->data[i] - operand2->data[i];
        }
        return output;
    }

    template <typename type>
    std::shared_ptr<tensor<type>> tensor_sub<type>::cpu_forward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> operand2)
    {
        return tensor_sub<type>::general_forward(operand1, operand2);
    }

    template <typename type>
    void tensor_sub<type>::general_backward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> output, std::shared_ptr<tensor<type>> operand2)
    {

        for (int i = 0; i < output->grad.size(); ++i)
        {
            operand1->grad[i] += output->grad[i];
            operand2->grad[i] += -1 * output->grad[i];
        }
    }

    template <typename type>
    void tensor_sub<type>::cpu_backward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> output, std::shared_ptr<tensor<type>> operand2)
    {
        return;
    }

} // namespace synaptic