#include "../../include/optimisers/gd.hpp"

namespace synaptic
{
    namespace optimisers
    {
        template <typename type>
        void gd<type>::step()
        {
            for(auto ele:this->optimisation_targets)
            {
                for(int i=0;i<ele->total;i++)
                {
                    ele->data[i] -= ele->grad[i]*this->learning_rate;
                }
            }
        }

        template <typename type>
        void gd<type>::zero_grad()
        {
            for(auto ele:this->optimisation_targets)
            {
                for(int i=0;i<ele->total;i++)
                {
                    ele->grad[i] = type(0);
                }
            }
        } 
    }
}