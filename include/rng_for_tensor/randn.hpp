#pragma once
#include "../abstracts/basic_op.hpp"
#include "../tensor.hpp"
#include <memory>
#include <functional>
#include <cstdlib>
#include <random>

namespace synaptic
{
    namespace rng_for_tensor
    {
        template <typename type>
        class randn
        {
        public:

            randn(float mean = 0,float standard_deviation = 1,devices dev = devices::none,unsigned int seed = time(NULL)) : device(dev),mean(mean),standard_deviation(standard_deviation){}
            float mean=0;
            float standard_deviation = 1;
            unsigned int seed = time(NULL);
            std::vector<int> output_shape;
            devices device = devices::none;

            std::shared_ptr<tensor<type>> generate(std::vector<int> shape);
        };
    }
}

#include "../src/rng_for_tensor/randn.tpp"
