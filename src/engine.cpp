#include <iostream>
#include <vector>
#include <memory>
#include <cassert>

enum class op {
    none,
    add,
    mul,
    sub,
    divi,
    pow,
    exp,
};

template <typename type>
class tensor {
public:
    std::vector<type> data;
    std::vector<std::shared_ptr<tensor<type>>> previous_nodes;
    std::vector<type> grad;
    std::vector<int> dims;
    int total;
    op operation = op::none;

    tensor() : total(1), dims({1}), data({0}), grad({0}) {}

    tensor(const std::vector<int>& shape) {
        total = 1;
        for (const auto& ele : shape) {
            total *= ele;
        }
        data.resize(total, 0);
        grad.resize(total, 0);
        dims = shape;
    }

    friend std::ostream& operator<<(std::ostream& output, const tensor& t) {
        output << "Tensor:\nData: \n";
        for (const auto& ele : t.data) {
            output << ele << " ";
        }
        output << "\nGrad : \n";
        for (const auto& ele : t.grad) {
            output << ele << " ";
        }
        return output;
    }

    static void add_backprop(std::shared_ptr<tensor>& a, std::shared_ptr<tensor>& b, const tensor& output) {
        for (int i = 0; i < output.grad.size(); ++i) {
            a->grad[i] += output.grad[i];
            b->grad[i] += output.grad[i];
        }
    }

    static std::shared_ptr<tensor> add(std::shared_ptr<tensor> a, std::shared_ptr<tensor> b) {
        assert(a->dims.size() == b->dims.size() && "Dimensions of tensors being added should be the same");
        bool shape_flag = true;
        for (int i = 0; i < a->dims.size(); ++i) {
            if (a->dims[i] != b->dims[i]) {
                shape_flag = false;
                break;
            }
        }
        assert(shape_flag && "Shape of Tensors not same");

        auto output = std::make_shared<tensor>(a->dims);
        output->operation = op::add;
        output->previous_nodes.push_back(a);
        output->previous_nodes.push_back(b);
        for (int i = 0; i < a->data.size(); ++i) {
            output->data[i] = a->data[i] + b->data[i];
        }
        return output;
    }

    static void recursive_backprop(std::shared_ptr<tensor> cur) {
        if (cur->operation == op::add) {
            add_backprop(cur->previous_nodes[0], cur->previous_nodes[1], *cur);
            if (cur->previous_nodes[0]->operation != op::none)
                recursive_backprop(cur->previous_nodes[0]);
            if (cur->previous_nodes[1]->operation != op::none)
                recursive_backprop(cur->previous_nodes[1]);
        }
    }

    void backprop() {
        this->grad = std::vector<type>(this->total, 1);
        recursive_backprop(std::make_shared<tensor>(*this));
    }
};

template <typename T>
std::shared_ptr<tensor<T>> operator+(std::shared_ptr<tensor<T>> a, std::shared_ptr<tensor<T>> b) {
    return tensor<T>::add(a, b);
}

int main() {
    auto t1 = std::make_shared<tensor<float>>(std::vector<int>{2});
    auto t2 = std::make_shared<tensor<float>>(std::vector<int>{2});
    t1->data = {1.0, 3.0};
    t2->data = {1.0, 4.0};

    auto res = t1 + t2;
    std::cout << *res << std::endl;
    std::cout << "Total of res: " << res->total << std::endl;
    std::cout << "Operation: " << static_cast<int>(res->operation) << std::endl;

    res->backprop();
    std::cout << "Backpropagation result:\n";
    std::cout << *t1 << std::endl;
    std::cout << *t2 << std::endl;

    return 0;
}