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
            gd(std::set<std::shared_ptr<tensor<type>>> &params,float lr = 0.001) : learning_rate(lr) ,optimisation_targets(params)
            {
                
            }

            float learning_rate=0.001;
            std::set<std::shared_ptr<tensor<type>>> optimisation_targets;

            void step();
            void zero_grad();
        };
    }
}

#include "../../src/optimisers/gd.tpp"