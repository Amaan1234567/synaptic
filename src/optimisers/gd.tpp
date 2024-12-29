#include "../../include/optimisers/gd.hpp"

namespace synaptic
{
    namespace optimisers
    {
        template <typename type>
        void gd<type>::step(std::set<std::shared_ptr<tensor<type>>> &optimisation_targets)
        {
            for(auto ele:optimisation_targets)
            {
                for(int i=0;i<ele->total;i++)
                {
                    ele->data[i] -= ele->grad[i]*this->learning_rate;
                }
            }
        }

        template <typename type>
        void gd<type>::zero_grad(std::set<std::shared_ptr<tensor<type>>> &optimisation_targets)
        {
            for(auto ele:optimisation_targets)
            {
                for(int i=0;i<ele->total;i++)
                {
                    ele->grad[i] = type(0);
                }
            }
        } 
    }
}