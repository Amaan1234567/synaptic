#include "../../include/operands/tensor_transpose.hpp"
#include "../../include/device_enum.hpp"

namespace synaptic
{

    template <typename type>
    typename tensor_transpose<type>::device_specific_forward tensor_transpose<type>::get_impl_selector_forward()
    {
        if (this->device == devices::cpu)
        {
            return [this](std::shared_ptr<tensor<type>> op1, std::shared_ptr<tensor<type>> op2)
            {
                return this->cpu_forward(op1, op2);
            };
        }
        // transpose checks for other devices (e.g., GPU) as necessary
        return [this](std::shared_ptr<tensor<type>> op1, std::shared_ptr<tensor<type>> op2)
        {
            return this->general_forward(op1, op2);
        };
    }

    template <typename type>
    typename tensor_transpose<type>::device_specific_backward tensor_transpose<type>::get_impl_selector_backward()
    {
        if (this->device == devices::cpu)
        {
            return [this](std::shared_ptr<tensor<type>> op1, std::shared_ptr<tensor<type>> grad, std::shared_ptr<tensor<type>> op2)
            {
                this->cpu_backward(op1, grad, op2);
            };
        }
        // transpose checks for other devices (e.g., GPU) as necessary
        return [this](std::shared_ptr<tensor<type>> op1, std::shared_ptr<tensor<type>> grad, std::shared_ptr<tensor<type>> op2)
        {
            this->general_backward(op1, grad, op2);
        };
    }

    template <typename type>
    std::shared_ptr<tensor<type>> tensor_transpose<type>::forward(std::shared_ptr<tensor<type>> operand1,std::shared_ptr<tensor<type>> operand2 )
    {
        // Get the appropriate implementation based on the current device
        auto impl = get_impl_selector_forward();
        // Call the selected implementation
        auto output = impl(operand1,operand2);
        return output;
    }

    template <typename type>
    void tensor_transpose<type>::backward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> output, std::shared_ptr<tensor<type>> operand2)
    {
        // Get the appropriate implementation based on the current device
        // std::cout << "Performing backward pass for tensor transposeition" << std::endl;
        auto impl = get_impl_selector_backward();
        // Call the selected implementation
        impl(operand1, output, operand2);
    }

    // The rest of the implementations (cpu_forward, cpu_backward, general_forward, general_backward)
    // remain the same as they were in your original code.

    template <typename type>
    std::shared_ptr<tensor<type>> tensor_transpose<type>::general_forward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> operand2)
    {
        if(this->dim0 == -1 & this->dim1 == -1)
        {
            this->dim0 = operand1->dims.size() - 1;
            this->dim1 = operand1->dims.size() - 2;
        }
        else if(this->dim0 == -1 && this->dim1 != -1 && this->dim1 != operand1->dims.size()-1)
        {
            this->dim0 = operand1->dims.size() - 1;
        }
        else if(this->dim0 != -1 && this->dim1 == -1 && this->dim0 != operand1->dims.size()-1)
        {
            this->dim1 = operand1->dims.size() - 1;
        }

        if (this->dim0 >= 0 && this->dim0 < operand1->dims.size() && this->dim1 >= 0 && this->dim1 < operand1->dims.size() && this->dim0 != this->dim1)
        {
            std::runtime_error("Please provide proper dimension indices, and make sure both the indices are different :)");
        }

        std::vector<int> output_dims = operand1->dims;
        std::swap(output_dims[this->dim0], output_dims[this->dim1]);

        auto output = std::make_shared<tensor<type>>(output_dims);
        auto dims_tensor = std::make_shared<synaptic::tensor<type>>(std::vector<int>{2});
        dims_tensor->data[0] = this->dim0;
        dims_tensor->data[1] = this->dim1;
        output->previous_nodes.push_back(operand1);
        output->previous_nodes.push_back(dims_tensor);
        output->operand_obj_ptr = std::make_shared<tensor_transpose<type>>(*this);

        // Calculate original strides
        std::vector<int> strides(operand1->dims.size(), 1);
        for(int i = operand1->dims.size() - 2; i >= 0; i--)
        {
            strides[i] = strides[i + 1] * operand1->dims[i + 1];
        }

        // Transposed strides
        std::swap(strides[this->dim0], strides[this->dim1]);

        // Perform the transpose by iterating through the output tensor
        std::vector<int> counts(operand1->dims.size(), 0); // Track the current multi-dimensional index
        int total_elements = output->total;

        for (int count = 0; count < total_elements; ++count)
        {
            int original_idx = 0;
            for (int i = 0; i < strides.size(); ++i)
            {
                original_idx += counts[i] * strides[i];
            }
            output->data[count] = operand1->data[original_idx];

            // Increment the multi-dimensional index
            for (int i = strides.size() - 1; i >= 0; --i)
            {
                counts[i]++;
                if (counts[i] < output->dims[i])
                {
                    break;
                }
                counts[i] = 0;
            }
        }

        return output;
    }

    template <typename type>
    std::shared_ptr<tensor<type>> tensor_transpose<type>::cpu_forward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> operand2)
    {
        return tensor_transpose<type>::general_forward(operand1, operand2);
    }

    template <typename type>
    void tensor_transpose<type>::general_backward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> output, std::shared_ptr<tensor<type>> operand2)
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
    void tensor_transpose<type>::cpu_backward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> output, std::shared_ptr<tensor<type>> operand2)
    {
        return;
    }

} // namespace synaptic