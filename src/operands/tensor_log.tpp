#include "../../include/operands/tensor_log.hpp"
#include "../../include/device_enum.hpp"

namespace synaptic
{

    template <typename type>
    typename tensor_log<type>::device_specific_forward tensor_log<type>::get_impl_selector_forward()
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
    typename tensor_log<type>::device_specific_backward tensor_log<type>::get_impl_selector_backward()
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
    std::shared_ptr<tensor<type>> tensor_log<type>::forward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> operand2)
    {
        // Get the appropriate implementation based on the current device
        auto impl = get_impl_selector_forward();
        // Call the selected implementation
        auto output = impl(operand1, operand2);
        return output;
    }

    template <typename type>
    void tensor_log<type>::backward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> output, std::shared_ptr<tensor<type>> operand2)
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
    std::shared_ptr<tensor<type>> tensor_log<type>::general_forward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> operand2)
    {
        std::cout << "base :"<<this->base <<std::endl;
        if(this->base <= type(0))
        {
            LOG_ERROR("tensor_log","base of logarithm can't be negative or 0!!");
            throw std::runtime_error("base of logarithm can't be negative or 0!!");
        }

        if(this->base == M_Ef64)
        this->log_of_base_precalculated = 1;
        else
        this->log_of_base_precalculated = std::log(base);
        
        auto base_storing_tensor = std::make_shared<tensor<float>>(std::vector<int>{1});
        base_storing_tensor->data[0] = base;
        auto output = std::make_shared<tensor<type>>(operand1->dims);
        output->operand_obj_ptr = std::make_shared<tensor_log<type>>(*this);
        output->previous_nodes.push_back(operand1);
        for (int i = 0; i < operand1->data.size(); ++i)
        {
            if(this->base == M_Ef64)
            output->data[i] = std::log(operand1->data[i]);
            else if(this->base == type(10))
            output->data[i] = std::log10(operand1->data[i]);
            else if(this->base == type(2))
            output->data[i] = std::log2(operand1->data[i]);
            else
            output->data[i] = std::log(operand1->data[i])/this->log_of_base_precalculated;
        }
        
        return output;
    }

    template <typename type>
    std::shared_ptr<tensor<type>> tensor_log<type>::cpu_forward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> operand2)
    {
        return tensor_log<type>::general_forward(operand1, operand2);
    }

    template <typename type>
    void tensor_log<type>::general_backward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> output, std::shared_ptr<tensor<type>> operand2)
    {
        std::cout<<"inside"<<std::endl;
        for (int i = 0; i < output->grad.size(); ++i)
        {
            operand1->grad[i] += (type(1)/(operand1->data[i]*this->log_of_base_precalculated)) * output->grad[i];
        }
    }

    template <typename type>
    void tensor_log<type>::cpu_backward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> output, std::shared_ptr<tensor<type>> operand2)
    {
        return;
    }

} // namespace synaptic