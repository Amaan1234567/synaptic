#include "../../include/connections/sigmoid.hpp"
#include "../../include/device_enum.hpp"
#include <cmath>

namespace synaptic
{
    namespace connections
    {
        template <typename type>
        typename sigmoid<type>::device_specific_forward sigmoid<type>::get_impl_selector_forward()
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
        typename sigmoid<type>::device_specific_backward sigmoid<type>::get_impl_selector_backward()
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
        std::shared_ptr<tensor<type>> sigmoid<type>::forward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> operand2)
        {
            // Get the appropriate implementation based on the current device
            auto impl = get_impl_selector_forward();
            // Call the selected implementation
            auto output = impl(operand1, operand2);
            return output;
        }

        template <typename type>
        void sigmoid<type>::backward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> output, std::shared_ptr<tensor<type>> operand2)
        {
            // Get the appropriate implementation based on the current device
            // std::cout << "Performing backward pass for tensor expition" << std::endl;
            auto impl = get_impl_selector_backward();
            // Call the selected implementation
            impl(operand1, output, operand2);
        }

        

        template <typename type>
        std::shared_ptr<tensor<type>> sigmoid<type>::general_forward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> operand2)
        {
            auto output = std::make_shared<tensor<type>>(operand1->dims);
            output->previous_nodes.push_back(operand1);
            output->operand_obj_ptr = std::make_shared<sigmoid<type>>(*this);

            for (int i = 0; i < operand1->total; i++)
            {
                    output->data[i] = type(1)/(type(1)+std::exp(-this->slope_param*operand1->data[i])); 
            }
            return output;
        }

        template <typename type>
        std::shared_ptr<tensor<type>> sigmoid<type>::cpu_forward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> operand2)
        {
            return sigmoid<type>::general_forward(operand1, operand2);
        }

        template <typename type>
        void sigmoid<type>::general_backward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> output, std::shared_ptr<tensor<type>> operand2)
        {
            for (int i = 0; i < output->total; i++)
            {
                operand1->grad[i] += this->slope_param * output->data[i]*(type(1) - output->data[i]) * output->grad[i];
                // std::cout<< *operand1 <<std::endl;
            }
        }

        template <typename type>
        void sigmoid<type>::cpu_backward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> output, std::shared_ptr<tensor<type>> operand2)
        {
            return general_backward(operand1,output,operand2);
        }

    }

} // namespace synaptic
