#pragma once

#include "../tensor.hpp"
#include <set>
#include <memory>

namespace synaptic
{
    namespace optimisers
    {   
        template <typename type>
        class gd
        {
            public:
            gd(float lr = 0.001): learning_rate(lr){};
            float learning_rate=0.001;

            void step(std::set<std::shared_ptr<tensor<type>>> &optimisation_targets);
            void zero_grad(std::set<std::shared_ptr<tensor<type>>> &optimisation_targets);
        };
    }
}

#include "../../src/optimisers/gd.tpp"