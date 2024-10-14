#include "../../include/connections/softmax.hpp"
#include "../../include/device_enum.hpp"
#include "tensor_exp.hpp"
#include <numeric>
#include <cmath>

namespace synaptic
{
    namespace connections
    {
        template <typename type>
        typename softmax<type>::device_specific_forward softmax<type>::get_impl_selector_forward()
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
        typename softmax<type>::device_specific_backward softmax<type>::get_impl_selector_backward()
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
        std::shared_ptr<tensor<type>> softmax<type>::forward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> operand2)
        {
            // Get the appropriate implementation based on the current device
            auto impl = get_impl_selector_forward();
            // Call the selected implementation
            auto output = impl(operand1, operand2);
            return output;
        }

        template <typename type>
        void softmax<type>::backward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> output, std::shared_ptr<tensor<type>> operand2)
        {
            // Get the appropriate implementation based on the current device
            // std::cout << "Performing backward pass for tensor expition" << std::endl;
            auto impl = get_impl_selector_backward();
            // Call the selected implementation
            impl(operand1, output, operand2);
        }

        template <typename type>
        std::shared_ptr<tensor<type>> softmax<type>::general_forward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> operand2)
        {
            if(this->dim<0)
            this->dim = operand1->dims.size()+this->dim;

            auto output = std::make_shared<tensor<type>>(operand1->dims);
            output->operand_obj_ptr = std::make_shared<softmax<type>>(*this);

            type max = *std::max_element(operand1->data.begin(), operand1->data.end());
            auto adjusted_operand = operand1 - max;

            auto exp = std::make_shared<tensor_exp<type>>();

            auto expd_tensor = exp->forward(adjusted_operand);

            std::vector<int> strides;

            int cur_total = 1;
            for (int i = operand1->dims.size() - 1; i >= 0; i--)
            {
                if (i == this->dim)
                {
                    cur_total = 1;
                }
                cur_total *= operand1->dims[i];
                strides.push_back(cur_total);
            }
            std::reverse(strides.begin(), strides.end());

            std::cout << "this->dim : "<<this->dim <<std::endl;
            

            std::vector<type> totals((operand1->total/operand1->dims[this->dim]),0);
            int totals_idx =0;
            int count=0;
            std::vector<int> counts(strides.size(),0);
            
        
            int dims_last_idx=counts.size()-1;

            type sum=type(0);
            
            while(count != operand1->total)
            {
                int idx = 0;

                for(int i=0;i<counts.size();i++)
                {
                    idx+=counts[i];
                }
                sum+=expd_tensor->data[idx];
                counts[this->dim]++;
                count++;
                if(counts[this->dim]==operand1->dims[this->dim])
                {
                    totals[totals_idx++]=sum;
                    sum=type(0);
                    for (int i = strides.size() - 1; i >= 0; --i)
                    {
                        counts[i]++;
                        if (counts[i] < operand1->dims[i])
                        {
                            if(i!=this->dim)
                            counts[this->dim]=0;
                            break;
                        }
                        counts[i] = 0;
                    }
                }

            }
            
            std::cout << "inside" << std::endl;
            
            count = 0;
            totals_idx=0;
            while(count != operand1->total)
            {
                int idx = 0;

                for(int i=0;i<counts.size();i++)
                {
                    idx+=counts[i];
                }
                output->data[idx]=expd_tensor->data[idx]/totals[totals_idx];
                counts[this->dim]++;
                count++;
                if(counts[this->dim]==operand1->dims[this->dim])
                {
                    totals_idx++;
                    for (int i = strides.size() - 1; i >= 0; --i)
                    {
                        counts[i]++;
                        if (counts[i] < operand1->dims[i])
                        {
                            if(i!=this->dim)
                            counts[this->dim]=0;
                            break;
                        }
                        counts[i] = 0;
                    }

                }

            }

            output->previous_nodes.push_back(operand1);

            return output;
        }

        template <typename type>
        std::shared_ptr<tensor<type>> softmax<type>::cpu_forward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> operand2)
        {
            return softmax<type>::general_forward(operand1, operand2);
        }

        template <typename type>
        void softmax<type>::general_backward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> output, std::shared_ptr<tensor<type>> operand2)
        {
            std::cout << "inside backprop" << std::endl;
            for (int i = 0; i < output->total; i++)
            {
                for (int j = 0; j < output->total; j++)
                {
                    if (i == j)
                        operand1->grad[i] += output->data[i] * (type(1) - output->data[j]) * output->grad[i];
                    else
                        operand1->grad[i] += -output->data[i] * output->data[j] * output->grad[i];
                }
                // std::cout<< *operand1 <<std::endl;
            }
        }

        template <typename type>
        void softmax<type>::cpu_backward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> output, std::shared_ptr<tensor<type>> operand2)
        {
            return general_backward(operand1, output, operand2);
        }

    }

} // namespace synaptic
