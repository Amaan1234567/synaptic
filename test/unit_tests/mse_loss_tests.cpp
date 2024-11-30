#include "../../include/synaptic.hpp"
#include <gtest/gtest.h> // Include the GoogleTest header
#include <memory>
#include <stdexcept>
#include <vector>

using namespace synaptic;
// Test case 1
TEST(TensorTest, MseOfTwoTensors1) {
  auto t1 = std::make_shared<tensor<float>>(std::vector<int>{5, 4});
  auto t2 = std::make_shared<tensor<float>>(std::vector<int>{5, 4});
  t1->data = {2.0153,  1.5694,  1.1120,  0.5215,  -0.0261, 1.1867,  0.4831,
              -1.4634, -1.4566, 0.1550,  0.1747,  -0.7770, -0.5263, -0.8394,
              -1.7548, -0.6452, -1.1722, -0.7855, -0.5609, 0.2995};
  t2->data = {1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
              1., 1., 1., 1., 1., 1., 1., 1., 1., 1.};

  auto mse = synaptic::loss_fn::mse<float>();
  // std::cout <<"inside" << std::endl;
  auto res = mse.forward(t1, t2);
  std::vector<float> expected = {2.3225};
  // std::cout<< res->total <<std::endl;
  for (int i = 0; i < res->total; i++) {
    EXPECT_NEAR(res->data[i], expected[i], 0.0001);
  }
}

TEST(TensorTest, MseOfTwoTensors2) {
  auto t1 = std::make_shared<tensor<float>>(std::vector<int>{16, 1});
  auto t2 = std::make_shared<tensor<float>>(std::vector<int>{16, 1});
  t1->data = {-0.2302, 2.0022,  0.9679,  0.7661,  0.3374,  -0.6907,
              0.2537,  -2.2319, -1.2492, -1.8190, -1.6991, 0.1217,
              1.5109,  -0.1092, 0.3973,  1.1184};
  t2->data = {0.1930,  2.5943,  -1.2468, -0.2796, -1.7764, -0.7096,
              -1.1037, 0.1378,  1.1170,  -0.8268, 0.0096,  0.5873,
              0.5555,  -0.5121, 0.1244,  -2.8887};

  auto mse = synaptic::loss_fn::mse<float>();
  // std::cout <<"inside" << std::endl;
  auto res = mse.forward(t1, t2);
  std::vector<float> expected = {2.8363};
  // std::cout<< res->total <<std::endl;
  for (int i = 0; i < res->total; i++) {
    EXPECT_NEAR(res->data[i], expected[i], 0.0001);
  }
}

TEST(TensorTest, MseOfTwoTensorsBackpropCheck) {
  auto t1 = std::make_shared<tensor<float>>(std::vector<int>{5, 4});
  auto t2 = std::make_shared<tensor<float>>(std::vector<int>{5, 4});
  t1->data = {2.0153,  1.5694,  1.1120,  0.5215,  -0.0261, 1.1867,  0.4831,
              -1.4634, -1.4566, 0.1550,  0.1747,  -0.7770, -0.5263, -0.8394,
              -1.7548, -0.6452, -1.1722, -0.7855, -0.5609, 0.2995};
  t2->data = {1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
              1., 1., 1., 1., 1., 1., 1., 1., 1., 1.};

  auto mse = synaptic::loss_fn::mse<float>();

  auto res = mse.forward(t1, t2);

  res->backprop();

  std::vector<float> expected1 = {0.1015,  0.0569,  0.0112,  -0.0478, -0.1026,
                                  0.0187,  -0.0517, -0.2463, -0.2457, -0.0845,
                                  -0.0825, -0.1777, -0.1526, -0.1839, -0.2755,
                                  -0.1645, -0.2172, -0.1786, -0.1561, -0.0700};

  std::vector<float> expected2 = {-0.1015, -0.0569, -0.0112, 0.0478, 0.1026,
                                  -0.0187, 0.0517,  0.2463,  0.2457, 0.0845,
                                  0.0825,  0.1777,  0.1526,  0.1839, 0.2755,
                                  0.1645,  0.2172,  0.1786,  0.1561, 0.0700};

  for (int i = 0; i < res->total; i++) {
    EXPECT_NEAR(t1->grad[i], expected1[i], 0.0001);
    EXPECT_NEAR(t2->grad[i], expected2[i], 0.0001);
  }
}

TEST(TensorTest, MseOfTwoTensorsBackpropCheck2) {
  auto t1 = std::make_shared<tensor<float>>(std::vector<int>{16, 1});
  auto t2 = std::make_shared<tensor<float>>(std::vector<int>{16, 1});
  t1->data = {-0.2302, 2.0022,  0.9679,  0.7661,  0.3374,  -0.6907,
              0.2537,  -2.2319, -1.2492, -1.8190, -1.6991, 0.1217,
              1.5109,  -0.1092, 0.3973,  1.1184};
  t2->data = {0.1930,  2.5943,  -1.2468, -0.2796, -1.7764, -0.7096,
              -1.1037, 0.1378,  1.1170,  -0.8268, 0.0096,  0.5873,
              0.5555,  -0.5121, 0.1244,  -2.8887};

  auto mse = synaptic::loss_fn::mse<float>();
  // std::cout <<"inside" << std::endl;
  auto res = mse.forward(t1, t2);

  res->backprop();

  std::vector<float> expected1 = {
      -0.0529, -0.0740, 0.2768,  0.1307,  0.2642, 0.0024, 0.1697, -0.2962,
      -0.2958, -0.1240, -0.2136, -0.0582, 0.1194, 0.0504, 0.0341, 0.5009};

  std::vector<float> expected2 = {
      0.0529, 0.0740, -0.2768, -0.1307, -0.2642, -0.0024, -0.1697, 0.2962,
      0.2958, 0.1240, 0.2136,  0.0582,  -0.1194, -0.0504, -0.0341, -0.5009};

  for (int i = 0; i < res->total; i++) {
    EXPECT_NEAR(t1->grad[i], expected1[i], 0.0001);
    EXPECT_NEAR(t2->grad[i], expected2[i], 0.0001);
  }
}

// Test cases where the shape mismatch triggers an assertion failure
TEST(TensorAssertionFailureTest, TensorMseShapeMismatch) {
  auto t1 = std::make_shared<tensor<float>>(std::vector<int>{2, 3});
  auto t2 = std::make_shared<tensor<float>>(std::vector<int>{3, 2});
  t1->data = {1, 2, 3, 4, 5, 6};
  t2->data = {1, 2, 3, 4, 5, 6};

  auto mse = synaptic::loss_fn::mse<float>();

  EXPECT_THROW(
      { std::shared_ptr<tensor<float>> res = mse.forward(t1, t2); },
      std::runtime_error);
}

TEST(TensorAssertionFailureTest, TensorMseWithScalarShapeMismatch) {
  auto t1 = std::make_shared<tensor<float>>(std::vector<int>{3});
  auto t2 = std::make_shared<tensor<float>>(std::vector<int>{1, 3});
  t1->data = {1.0f, 2.0f, 3.0f};
  t2->data = {1.0f, 2.0f, 3.0f};

  auto mse = synaptic::loss_fn::mse<float>();

  EXPECT_THROW({ auto res = mse.forward(t1, t2); }, std::runtime_error);
}