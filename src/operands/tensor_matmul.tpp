#include "../../include/operands/tensor_matmul.hpp"
#include "../../include/operands/tensor_transpose.hpp"
#include "../../include/device_enum.hpp"

namespace synaptic
{

    template <typename type>
    typename tensor_matmul<type>::device_specific_forward tensor_matmul<type>::get_impl_selector_forward()
    {
        if (this->device == devices::cpu)
        {
            return [this](std::shared_ptr<tensor<type>> op1, std::shared_ptr<tensor<type>> op2)
            {
                return this->cpu_forward(op1, op2);
            };
        }
        // matmul checks for other devices (e.g., GPU) as necessary
        return [this](std::shared_ptr<tensor<type>> op1, std::shared_ptr<tensor<type>> op2)
        {
            return this->general_forward(op1, op2);
        };
    }

    template <typename type>
    typename tensor_matmul<type>::device_specific_backward tensor_matmul<type>::get_impl_selector_backward()
    {
        if (this->device == devices::cpu)
        {
            return [this](std::shared_ptr<tensor<type>> op1, std::shared_ptr<tensor<type>> grad, std::shared_ptr<tensor<type>> op2)
            {
                this->cpu_backward(op1, grad, op2);
            };
        }
        // matmul checks for other devices (e.g., GPU) as necessary
        return [this](std::shared_ptr<tensor<type>> op1, std::shared_ptr<tensor<type>> grad, std::shared_ptr<tensor<type>> op2)
        {
            this->general_backward(op1, grad, op2);
        };
    }

    template <typename type>
    std::shared_ptr<tensor<type>> tensor_matmul<type>::forward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> operand2)
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
    void tensor_matmul<type>::backward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> output, std::shared_ptr<tensor<type>> operand2)
    {
        // Get the appropriate implementation based on the current device
        // std::cout << "Performing backward pass for tensor matmulition" << std::endl;
        auto impl = get_impl_selector_backward();
        // Call the selected implementation
        impl(operand1, output, operand2);
    }

    // The rest of the implementations (cpu_forward, cpu_backward, general_forward, general_backward)
    // remain the same as they were in your original code.

    template <typename type>
    void tensor_matmul<type>::matmul_general_impl(std::shared_ptr<synaptic::tensor<type>> operand1, std::shared_ptr<synaptic::tensor<type>> operand2, std::shared_ptr<synaptic::tensor<type>> output, std::vector<int> &custom_dims_a, std::vector<int> &custom_dims_b)
    {
        for (int batch = 0; batch < custom_dims_a[0]; batch++)
        {
            for (int i = 0; i < custom_dims_a[1]; i++)
            {
                for (int j = 0; j < custom_dims_b[2]; j++)
                {
                    int output_idx = batch * (custom_dims_a[1] * custom_dims_b[2]) + i * custom_dims_b[2] + j;
                    for (int k = 0; k < custom_dims_a[2]; k++)
                    {
                        int operand1_idx = batch * (custom_dims_a[1] * custom_dims_a[2]) + i * custom_dims_a[2] + k;
                        int operand2_idx = batch * (custom_dims_b[1] * custom_dims_b[2]) + k * custom_dims_b[2] + j;
                        output->data[output_idx] += operand1->data[operand1_idx] * operand2->data[operand2_idx];
                        // std::cout << output_idx << " " << operand1_idx << " " << operand2_idx << std::endl;
                    }
                }
                // std::cout << std::endl;
            }
        }
    }

    template <typename type>
    bool tensor_matmul<type>::dim_check(std::shared_ptr<synaptic::tensor<type>> operand1, std::shared_ptr<synaptic::tensor<type>> operand2)
    {
        if (operand1->dims.size() != operand2->dims.size())
        {
            throw std::runtime_error("Number of Dimensions of tensors being added should be the same");
        }
        return true;
    }

    template <typename type>
    std::shared_ptr<tensor<type>> tensor_matmul<type>::general_forward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> operand2)
    {
        tensor_matmul<type>::dim_check(operand1, operand2);
        bool matmul_flag = true;
        auto last_dim = operand1->dims.size() - 1;
        std::cout << operand1->dims[last_dim] << " " << operand2->dims[last_dim - 1] << std::endl;
        if (operand1->dims[last_dim] != operand2->dims[last_dim - 1])
            throw std::runtime_error("Tensors are not compatible for matrix multiplication please check shapes");

        std::vector<int> custom_dims_operand1(3, 0);
        std::vector<int> custom_dims_operand2(3, 0);
        custom_dims_operand1[2] = operand1->dims[operand1->dims.size() - 1];
        custom_dims_operand1[1] = operand1->dims[operand1->dims.size() - 2];
        custom_dims_operand2[2] = operand2->dims[operand2->dims.size() - 1];
        custom_dims_operand2[1] = operand2->dims[operand2->dims.size() - 2];

        int total1 = 1;
        int total2 = 1;
        for (int i = 0; i < operand1->dims.size() - 2; i++)
        {
            total1 *= operand1->dims[i];
            total2 *= operand2->dims[i];
        }
        if (total1 == total2)
        {
            std::runtime_error("Tensors do not have same shape");
        }

        custom_dims_operand1[0] = total1;
        custom_dims_operand2[0] = total2;

        std::vector<int> output_dims(operand1->dims.begin(), operand1->dims.end());
        output_dims[output_dims.size() - 1] = operand2->dims[operand2->dims.size() - 1];
        std::cout << "output dims: ";
        for (auto dims : output_dims)
        {
            std::cout << dims << " ";
        }
        std::cout << std::endl;
        auto output = std::make_shared<tensor<type>>(output_dims);
        output->operand_obj_ptr = std::make_shared<tensor_matmul<type>>(*this);
        output->previous_nodes.push_back(operand1);
        output->previous_nodes.push_back(operand2);

        tensor_matmul<type>::matmul_general_impl(operand1, operand2, output, custom_dims_operand1, custom_dims_operand2);
        return output;
    }

    template <typename type>
    std::shared_ptr<tensor<type>> tensor_matmul<type>::cpu_forward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> operand2)
    {
        return tensor_matmul<type>::general_forward(operand1, operand2);
    }

    template <typename type>
    void tensor_matmul<type>::general_backward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> output, std::shared_ptr<tensor<type>> operand2)
    {
        auto grad = std::make_shared<tensor<type>>(output->dims);
        grad->data = output->grad;

        auto transpose_impl1 = std::make_shared<tensor_transpose<type>>(this->device,operand2->dims.size() - 1,operand2->dims.size() - 2);
        auto transpose_impl2 = std::make_shared<tensor_transpose<type>>(this->device,operand2->dims.size() - 2,operand2->dims.size() - 1);
        

        auto res1 = tensor_matmul<type>::forward(grad, transpose_impl1->forward(operand2));
        auto res2 = tensor_matmul<type>::forward(transpose_impl2->forward(operand1), grad);

        operand1->grad = res1->data;
        operand2->grad = res2->data;
    }

    template <typename type>
    void tensor_matmul<type>::cpu_backward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> output, std::shared_ptr<tensor<type>> operand2)
    {
        return;
    }

} // namespace synaptic