#include "../../include/connections/silu.hpp"
#include "../../include/connections/sigmoid.hpp"
#include "../../include/device_enum.hpp"
#include <cmath>

namespace synaptic
{
    namespace connections
    {
        template <typename type>
        typename silu<type>::device_specific_forward silu<type>::get_impl_selector_forward()
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
        typename silu<type>::device_specific_backward silu<type>::get_impl_selector_backward()
        {
            if (this->device == devices::cpu)
            {
                return [this](std::shared_ptr<tensor<type>> op1, std::shared_ptr<tensor<type>> grad, std::shared_ptr<tensor<type>> op2)
                {
                    this->cpu_backward(op1, grad, op2);
                };
            }
            std::cout<<"running"<<std::endl;
            // exp checks for other devices (e.g., GPU) as necessary
            return [this](std::shared_ptr<tensor<type>> op1, std::shared_ptr<tensor<type>> grad, std::shared_ptr<tensor<type>> op2)
            {
                this->general_backward(op1, grad, op2);
            };
        }

        template <typename type>
        std::shared_ptr<tensor<type>> silu<type>::forward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> operand2)
        {
            // Get the appropriate implementation based on the current device
            auto impl = this->get_impl_selector_forward();
            // Call the selected implementation
            auto output = impl(operand1, operand2);
            return output;
        }

        template <typename type>
        void silu<type>::backward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> output, std::shared_ptr<tensor<type>> operand2)
        {
            // Get the appropriate implementation based on the current device
            std::cout << "Performing backward pass for tensor expition" << std::endl;
            auto impl = this->get_impl_selector_backward();
            // Call the selected implementation
            impl(operand1, output, operand2);
        }

        

        template <typename type>
        std::shared_ptr<tensor<type>> silu<type>::general_forward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> operand2)
        {
            auto sig = std::make_shared<sigmoid<type>>();
            auto output = operand1 * sig->forward(operand1);
            //output->operand_obj_ptr = std::make_shared<silu<type>>(*this);


            return output;
        }

        template <typename type>
        std::shared_ptr<tensor<type>> silu<type>::cpu_forward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> operand2)
        {
            return silu<type>::general_forward(operand1, operand2);
        }

        template <typename type>
        void silu<type>::general_backward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> output, std::shared_ptr<tensor<type>> operand2)
        {
            //left blank cause the forward was composed of pre existing operations, so the backward is handled automatically when backprop is called
        }

        template <typename type>
        void silu<type>::cpu_backward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> output, std::shared_ptr<tensor<type>> operand2)
        {
            return general_backward(operand1,output,operand2);
        }

    }

} // namespace synaptic
