#pragma once
#include "basic_op.hpp"
#include <set>
#include <memory>
namespace synaptic
{
    namespace layers
    {
        template <typename type>
        class module 
        {
            public:
            std::set<std::shared_ptr<tensor<type>>> optimisation_targets;
            // Forward function definition
            virtual std::shared_ptr<tensor<type>> forward(std::shared_ptr<tensor<type>> input) {
                // Provide a default or pure virtual definition (e.g., throw an error if not overridden)
                throw std::runtime_error("forward() is not implemented for this module.");
            }

            // Register modules function definition
            virtual void register_modules() {
                
            }
            //add all the modules being used in model to optimisation_targets using this function
        };
    }
}