#pragma once
#include "../abstracts/basic_op.hpp"
#include "../tensor.hpp"
#include <memory>
#include <functional>
#include <cstdlib>

namespace synaptic
{
    namespace rng_for_tensor
    {
        template <typename type>
        class randn
        {
        public:

            randn(std::vector<int> shape,float lower_limit=0,float higher_limit=1,devices dev = devices::none,unsigned int seed = time(NULL)) : device(dev),lower_limit(lower_limit),higher_limit(higher_limit),output_shape(shape){}
            float lower_limit = 0;
            float higher_limit = 0;
            unsigned int seed = time(NULL);
            std::vector<int> output_shape;
            devices device = devices::none;

            std::shared_ptr<tensor<type>> generate(std::vector<int> shape);
        };
    }
}

#include "../src/rng_for_tensor/randn.tpp"
