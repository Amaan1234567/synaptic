#include "../../include/rng_for_tensor/randn.hpp"
namespace synaptic
{
    namespace rng_for_tensor
    {
        template <typename type>
        std::shared_ptr<tensor<type>> randn<type>::generate(std::vector<int> shape)
        {
            std::default_random_engine generator(this->seed);
            std::cout <<"SD: "<< std::to_string(this->standard_deviation)<<std::endl;
            std::normal_distribution<double> distribution(this->mean, this->standard_deviation);
            auto data = std::make_shared<tensor<type>>(shape);
            for(auto &ele: data->data)
            {
                ele = distribution(generator);
            } 
            return data;
        }
    }
}