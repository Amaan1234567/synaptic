#include "connections.hpp"

namespace connections
{

    template <typename type>
    class relu
    {
    public:
        relu(type val) : non_linearity_multiplier(val) {}
        relu(type val1, type val2) : non_linearity_multiplier(val1), below_thres_value(val2) {}

        type non_linearity_multiplier = type(1);
        type below_thres_value = type(0);

        std::shared_ptr<synaptic::tensor<type>> forward(const std::shared_ptr<synaptic::tensor<type>> &a);
    };

}