#include "../../include/rng_for_tensor/randn.hpp"
namespace synaptic
{
    namespace rng_for_tensor
    {
        template <typename type>
        std::shared_ptr<tensor<type>> randn<type>::generate(std::vector<int> shape)
        {
            srand(this->seed);
            auto data = std::make_shared<tensor<type>>(shape);
            type range = (type)(this->higher_limit-this->lower_limit);
            for(auto &ele: data->data)
            {
                ele = (type)((float)rand()/RAND_MAX)*(range)+this->lower_limit;
            } 
            return data;
        }
    }
}