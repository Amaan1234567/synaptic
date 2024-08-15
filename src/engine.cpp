#include <iostream>
#include <vector>
#include <set>
#include <functional>
#include <cassert>

template <typename type>
class tensor
{
    public:
        std::vector<type> data;
        std::set<tensor> previous_nodes;
        std::vector<type> grad;
        std::vector<int> dims;
        std::function<void()> backward;
    tensor()
    {
        data = {0};
        grad = {0};
        dims = {1};
    }

    tensor(std::vector<int>& shape)
    {
        int total=1;
        for(auto &ele:shape)
        {
            std::cout<<ele<<std::endl;
            total*=ele;
        }
        std::cout<<"total: "<<total<<std::endl;
        data.resize(total);
        grad.resize(total);
        dims = shape;
    }

    friend std::ostream& operator<<(std::ostream& output,tensor t)
    {
        output << "Tensor:\n";
        output << "Data: \n";
        for(type ele: t.data)
        {
            output << ele << " ";
        }
        output << std::endl;
        output << "Grad : \n";
        for(type ele: t.grad)
        {
            output << ele << " ";
        }
        return output;
    }

    friend tensor operator+(tensor& a,tensor& b)
    {
        assert((a.dims.size() == b.dims.size(), "Dimensions of tensors being added should be same"));
        bool shape_flag=true;
        for(int i=0;i<a.dims.size();i++)
        {
            if(a.dims[i]!=b.dims[i])
            {
                shape_flag = false;
                break;
            }
        }
        assert((shape_flag==false, "Shape of Tensors not same"));
        tensor output = tensor<type>(a.dims);
        for(int i=0;i<a.data.size();i++)
        {
            output.data[i]=a.data[i]+b.data[i];
        }
        return output;
    }
    
};



int main()
{
    std::cout<<"hello there\n";
    tensor<float> t1,t2;
    t2.data = {1.0,3.0};
    t1.dims[0] = 2;
    t2.dims[0] = 2;
    t1.data = {1.0,3.0};
    //std::cout<<t1<<std::endl;
    //std::cout<<t2<<std::endl;
    std::cout<<t1+t2<<std::endl;
    return 0;
}