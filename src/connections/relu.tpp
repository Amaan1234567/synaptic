#include "../../include/connections/relu.hpp"
#include "../../include/device_enum.hpp"

namespace synaptic
{
    namespace connections
    {
        template <typename type>
        typename relu<type>::device_specific_forward relu<type>::get_impl_selector_forward()
        {
            if (this->device == devices::cpu)
            {
                return [this](std::shared_ptr<tensor<type>> op1, std::shared_ptr<tensor<type>> op2)
                {
                    return this->cpu_forward(op1, op2);
                };
            }
            // exp checks for other devices (e.g., GPU) as necessary
            return [this](std::shared_ptr<tensor<type>> op1, std::shared_ptr<tensor<type>> op2)
            {
                return this->general_forward(op1, op2);
            };
        }

        template <typename type>
        typename relu<type>::device_specific_backward relu<type>::get_impl_selector_backward()
        {
            if (this->device == devices::cpu)
            {
                return [this](std::shared_ptr<tensor<type>> op1, std::shared_ptr<tensor<type>> grad, std::shared_ptr<tensor<type>> op2)
                {
                    this->cpu_backward(op1, grad, op2);
                };
            }
            // exp checks for other devices (e.g., GPU) as necessary
            return [this](std::shared_ptr<tensor<type>> op1, std::shared_ptr<tensor<type>> grad, std::shared_ptr<tensor<type>> op2)
            {
                this->general_backward(op1, grad, op2);
            };
        }

        template <typename type>
        std::shared_ptr<tensor<type>> relu<type>::forward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> operand2)
        {
            // Get the appropriate implementation based on the current device
            auto impl = get_impl_selector_forward();
            // Call the selected implementation
            auto output = impl(operand1, operand2);
            return output;
        }

        template <typename type>
        void relu<type>::backward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> output, std::shared_ptr<tensor<type>> operand2)
        {
            // Get the appropriate implementation based on the current device
            // std::cout << "Performing backward pass for tensor expition" << std::endl;
            auto impl = get_impl_selector_backward();
            // Call the selected implementation
            impl(operand1, output, operand2);
        }

        // The rest of the implementations (cpu_forward, cpu_backward, general_forward, general_backward)
        // remain the same as they were in your original code.

        template <typename type>
        std::shared_ptr<tensor<type>> relu<type>::general_forward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> operand2)
        {
            auto output = std::make_shared<tensor<type>>(operand1->dims);
            output->previous_nodes.push_back(operand1);
            output->operand_obj_ptr = std::make_shared<relu<type>>(*this);

            for (int i = 0; i < operand1->total; i++)
            {
                if (operand1->data[i] > type(0))
                    output->data[i] = operand1->data[i] * this->non_linearity_multiplier;
                else
                    output->data[i] = this->below_thres_value; // Ensure uninitialized data is handled
            }
            return output;
        }

        template <typename type>
        std::shared_ptr<tensor<type>> relu<type>::cpu_forward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> operand2)
        {
            return relu<type>::general_forward(operand1, operand2);
        }

        template <typename type>
        void relu<type>::general_backward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> output, std::shared_ptr<tensor<type>> operand2)
        {
            for (int i = 0; i < output->total; i++)
            {
                type grad_multiplier = (operand1->data[i] > type(0))
                                           ? this->non_linearity_multiplier
                                           : this->below_thres_value;

                operand1->grad[i] += grad_multiplier * output->grad[i];
                // std::cout<< *operand1 <<std::endl;
            }
        }

        template <typename type>
        void relu<type>::cpu_backward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> output, std::shared_ptr<tensor<type>> operand2)
        {
            return;
        }

    }

} // namespace synaptic
