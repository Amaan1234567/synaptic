#pragma once
#include "../abstracts/basic_op.hpp"
#include "../rng_for_tensor/randn.hpp"
#include "../tensor.hpp"
#include <memory>
#include <functional>

namespace synaptic
{
    namespace layers
    {
        template <typename type>
        class linear : public basic_op<type>
        {
        public:

            linear(int in_features,int out_features, devices dev = devices::none) : device(dev) 
            {
                auto randn = synaptic::rng_for_tensor::randn<type>(0,std::sqrt(2.0/in_features));
                this->weights = randn.generate(std::vector<int>{in_features,out_features});
                this->biases = randn.generate(std::vector<int>{1,out_features});

                /* this->weights = std::make_shared<tensor<type>>(std::vector<int>{in_features,out_features});
                this->weights->data = std::vector<type>(this->weights->total,0);
                this->biases = std::make_shared<tensor<type>>(std::vector<int>{1,out_features});
                this->biases->data = std::vector<type>(this->biases->total,0); */
                std::cout << "weights of linear layer :"<<*this->weights<<std::endl;
                std::cout << "biases of linear layer :"<<*this->biases<<std::endl;
                // basically have the weights of the neurons in each column, no need of transposing anything
                //with weights initialised randomly
            }
            std::shared_ptr<tensor<type>> weights;
            std::shared_ptr<tensor<type>> biases;
            devices device = devices::none;
            

            // Use std::function instead of function pointer
            using device_specific_forward = std::function<std::shared_ptr<tensor<type>>(std::shared_ptr<tensor<type>>, std::shared_ptr<tensor<type>>)>;
            using device_specific_backward = std::function<void(std::shared_ptr<tensor<type>>, std::shared_ptr<tensor<type>>, std::shared_ptr<tensor<type>>)>;

            device_specific_forward get_impl_selector_forward();
            device_specific_backward get_impl_selector_backward();

            std::shared_ptr<tensor<type>> forward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> operand2 = nullptr);
            void backward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> output, std::shared_ptr<tensor<type>> operand2 = nullptr);

        private:
            std::shared_ptr<tensor<type>> cpu_forward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> operand2);
            void cpu_backward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> output, std::shared_ptr<tensor<type>> operand2 = nullptr);
            std::shared_ptr<tensor<type>> general_forward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> operand2);
            void general_backward(std::shared_ptr<tensor<type>> operand1, std::shared_ptr<tensor<type>> output, std::shared_ptr<tensor<type>> operand2 = nullptr);
        };
    }
}

#include "../src/layers/linear.tpp"
