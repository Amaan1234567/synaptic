#pragma once
#include "../abstracts/basic_op.hpp"
#include "../tensor.hpp"
#include <memory>
#include <functional>
#include <cstdlib>
#include <random>
#include <chrono>

namespace synaptic
{
    namespace rng_for_tensor
    {
        template <typename type>
        class randn
        {
        public:

            randn(float mean = 0,float standard_deviation = 1.0,devices dev = devices::none,unsigned int seed = std::random_device{}()) : device(dev),mean(mean),standard_deviation(standard_deviation),seed(seed){}
            float mean=0.0;
            float standard_deviation = 1.0;
            unsigned int seed = std::random_device{}();
            std::vector<int> output_shape;
            devices device = devices::none;

            std::shared_ptr<tensor<type>> generate(std::vector<int> shape);
        };
    }
}

#include "../src/rng_for_tensor/randn.tpp"
