#include "../../include/rng_for_tensor/randn.hpp"
namespace synaptic
{
    namespace rng_for_tensor
    {
        template <typename type>
        std::shared_ptr<tensor<type>> randn<type>::generate(std::vector<int> shape)
        {
            srand(this->seed);
            std::default_random_engine generator;
            std::normal_distribution<double> distribution(this->mean,this->standard_deviation);
            auto data = std::make_shared<tensor<type>>(shape);
            for(auto &ele: data->data)
            {
                ele = (type)(distribution(generator));
            } 
            return data;
        }
    }
}