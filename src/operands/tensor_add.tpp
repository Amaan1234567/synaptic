#include "../../include/operands/tensor_add.hpp"
#include "../../include/device_enum.hpp"

namespace synaptic
{

    template <typename type>
    typename tensor_add<type>::device_specific_forward tensor_add<type>::get_impl_selector_forward()
    {
        if (this->device == devices::cpu)
        {
            return [this](std::shared_ptr<tensor<type>> op1, std::shared_ptr<tensor<type>> op2)
            {
                return this->cpu_forward(op1, op2);
            };
        }
        // Add checks for other devices (e.g., GPU) as necessary
        return [this](std::shared_ptr<tensor<type>> op1, std::shared_ptr<tensor<type>> op2)
        {
            return this->general_forward(op1, op2);
        };
    }

    template <typename type>
    typename tensor_add<type>::device_specific_backward tensor_add<type>::get_impl_selector_backward()
    {
        if (this->device == devices::cpu)
        {
            return [this](std::shared_ptr<tensor<type>> op1, std::shared_ptr<tensor<type>> grad, std::shared_ptr<tensor<type>> op2)
            {
                this->cpu_backward(op1, grad, op2);
            };
        }
        // Add checks for other devices (e.g., GPU) as necessary
        return [this](std::shared_ptr<tensor<type>> op1, std::shared_ptr<tensor<type>> grad, std::shared_ptr<tensor<type>> op2)
        {
            this->general_backward(op1, grad, op2);
        };
    }

    template <typename type>
    std::shared_ptr<tensor<type>> tensor_add<type>::forward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> operand2)
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
    void tensor_add<type>::backward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> output, std::shared_ptr<tensor<type>> operand2)
    {
        // Get the appropriate implementation based on the current device
        // std::cout << "Performing backward pass for tensor addition" << std::endl;
        auto impl = get_impl_selector_backward();
        // Call the selected implementation
        impl(operand1, output, operand2);
    }

    // The rest of the implementations (cpu_forward, cpu_backward, general_forward, general_backward)
    // remain the same as they were in your original code.

    template <typename type>
    std::shared_ptr<tensor<type>> tensor_add<type>::general_forward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> operand2)
    {
        // Perform compatibility tests
        std::shared_ptr<tensor<type>> output;
        synaptic::tensor<type>::common_tensor_compatibility_tests(operand1, operand2);
        this->batch=false;
        if(operand1->dims[0]!=operand2->dims[0] && (operand1->dims[0]==1 || operand2->dims[0]==1 ))
        this->batch=true;
        if(this->batch)
        {
            if(operand1->dims[0]>1)
            output = std::make_shared<tensor<type>>(operand1->dims);
            else
            auto output = std::make_shared<tensor<type>>(operand2->dims); 

        }
        else
        output = std::make_shared<tensor<type>>(operand2->dims);
        
        output->previous_nodes.push_back(operand1);
        output->previous_nodes.push_back(operand2);
        output->operand_obj_ptr = std::make_shared<tensor_add<type>>(*this);
        // output->operand_obj_ptr = this;

        // std::cout << "address of new pointer created: "<< output->operand_obj_ptr << std::endl;
        //  Assuming both operands have the same size
        if(this->batch==false)
        {
            for (size_t i = 0; i < operand1->data.size(); ++i)
            {
                output->data[i] = operand1->data[i] + operand2->data[i];
            }
            return output;
        }
        else
        {   
            //temp need to fix this, make it work for all types of tensors
            if(operand2->dims[0]==1)
            {
                for(int batch=0;batch<operand1->dims[0];batch++)
                {
                    for (size_t i = 0; i < operand1->dims[1]; ++i)
                    {
                        output->data[operand1->dims[1]*batch+i] = operand1->data[operand1->dims[1]*batch+i] + operand2->data[i];
                    }
                }
            }
            else
            {
                for(int batch=0;batch<operand2->dims[0];batch++)
                {
                    for (size_t i = 0; i < operand2->dims[1]; ++i)
                    {
                        output->data[operand2->dims[1]*batch+i] = operand1->data[i] + operand2->data[operand2->dims[1]*batch+i];
                    }
                }
            }  
        }
        std::cout <<"output in tensor_add: "<< *output<<std::endl;
        return output;
    }

    template <typename type>
    std::shared_ptr<tensor<type>> tensor_add<type>::cpu_forward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> operand2)
    {
        return tensor_add<type>::general_forward(operand1, operand2);
    }

    template <typename type>
    void tensor_add<type>::general_backward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> output, std::shared_ptr<tensor<type>> operand2)
    {
        // Implement the backward pass for CPU
        // This is a placeholder implementation and should be replaced with actual logic
        if(this->batch)
        {   
            std::cout <<"inside tensor_add, batch variant backprop"<<std::endl;
            if(operand2->dims[0]==1)
            {
                for (size_t i = 0; i < operand1->total; ++i)
                {
                    operand1->grad[i] += output->grad[i];
                    //std::cout << i%(operand2->dims[1])<<" "<<output->grad[i] << std::endl;
                    operand2->grad[i%(operand2->dims[1])] += output->grad[i];
                }
            }
            else
            {
                for (size_t i = 0; i < operand1->total; ++i)
                {
                    operand2->grad[i] += output->grad[i];
                    operand1->grad[i%operand1->dims[1]] += output->grad[i];
                }
            }
        }
        else
        {
            // std::cout << "Performing backward pass for tensor addition on general" << std::endl;
            for (size_t i = 0; i < output->data.size(); ++i)
            {
                operand1->grad[i] += output->grad[i];
                operand2->grad[i] += output->grad[i];
            }
        }
        std::cout <<"op1 grad: "<< *operand1 << std::endl;
        std::cout <<"op2 grad: "<< *operand2 << std::endl;
        
    }

    template <typename type>
    void tensor_add<type>::cpu_backward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> output, std::shared_ptr<tensor<type>> operand2)
    {
        return;
    }

} // namespace synaptic