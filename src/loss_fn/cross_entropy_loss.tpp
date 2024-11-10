#include "../../include/loss_fn/cross_entropy_loss.hpp"
#include "../../include/device_enum.hpp"
#include "../../include/operands/tensor_pow.hpp"
#include <cmath>

namespace synaptic
{
    namespace loss_fn
    {
        template <typename type>
        typename cross_entropy_loss<type>::device_specific_forward cross_entropy_loss<type>::get_impl_selector_forward()
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
        typename cross_entropy_loss<type>::device_specific_backward cross_entropy_loss<type>::get_impl_selector_backward()
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
        std::shared_ptr<tensor<type>> cross_entropy_loss<type>::forward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> operand2)
        {
            // Get the appropriate implementation based on the current device
            auto impl = this->get_impl_selector_forward();
            // Call the selected implementation
            auto output = impl(operand1, operand2);
            return output;
        }

        template <typename type>
        void cross_entropy_loss<type>::backward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> output, std::shared_ptr<tensor<type>> operand2)
        {
            // Get the appropriate implementation based on the current device
            std::cout << "Performing backward pass for cross_entropy_loss" << std::endl;
            auto impl = this->get_impl_selector_backward();
            // Call the selected implementation
            impl(operand1, output, operand2);
        }

        

        template <typename type>
        std::shared_ptr<tensor<type>> cross_entropy_loss<type>::general_forward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> operand2)
        {
            //operand1:prediction  operand2:expected
            synaptic::tensor<type>::common_tensor_compatibility_tests(operand1, operand2);
            //std::cout << "inside" <<std::endl;
            
            auto softmax = std::make_shared<synaptic::connections::softmax<type>>(operand1->dims.size()-1);

            auto softmax_output = softmax->forward(operand1);

            
            auto log_of_expected = tensor<type>::log(softmax_output);

            auto output = std::make_shared<tensor<type>>(std::vector<int>{1});

            type sum = type(0);

            for(int i=0;i<operand1->total;i++)
            sum += operand2->data[i]*log_of_expected->data[i];

            std::cout << "sum: "<<sum << std::endl;
            if(this->reduction == "mean")
            output->data[0] = (-sum)/(type(operand1->dims[0]));

            this->softmaxed_data_store = softmax_output;
            output->operand_obj_ptr = std::make_shared<cross_entropy_loss<type>>(*this);
            
            output->previous_nodes.push_back(operand1);
            output->previous_nodes.push_back(operand2);
            return output;
        }

        template <typename type>
        std::shared_ptr<tensor<type>> cross_entropy_loss<type>::cpu_forward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> operand2)
        {
            return cross_entropy_loss<type>::general_forward(operand1, operand2);
        }

        template <typename type>
        void cross_entropy_loss<type>::general_backward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> output, std::shared_ptr<tensor<type>> operand2)
        {
            std::cout <<"softmax_output: "<<*operand1 <<std::endl;
            auto res = (this->softmaxed_data_store-operand2)/type(operand1->dims[0]);
            for(int i =0; i<operand1->total;i++)
            {
                operand1->grad[i] += res->data[i] * output->grad[i];
            }
        }

        template <typename type>
        void cross_entropy_loss<type>::cpu_backward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> output, std::shared_ptr<tensor<type>> operand2)
        {
            return general_backward(operand1,output,operand2);
        }

    }

} // namespace synaptic
