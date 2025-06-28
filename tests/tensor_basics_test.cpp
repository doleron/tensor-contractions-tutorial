#include <random>

#include <gtest/gtest.h>

#include "tensor.hpp"
#include "basic_tests_defs.hpp"

const static float EPISLON = 1e-7f;

TEST(Tensor, TensorInitialization_rank0)
{
    BEGIN_TEST()

    auto A = Tensor<float>({}, {9});

    EXPECT_NEAR(9, A.get({0}), EPISLON) << "index lookup faulire";

    END_TEST();
}

TEST(Tensor, TensorInitialization_rank1)
{
    BEGIN_TEST()

    auto A = Tensor<float>({4}, {0, 1, 2, 3});
    int expected[4] = {0, 1, 2, 3};

    int p = 0;
    for (int i = 0; i < 2; ++i)
    {
        EXPECT_NEAR(expected[p++], A.get({i}), EPISLON) << "index lookup faulire";
    }

    END_TEST();
}

TEST(Tensor, TensorInitialization_rank2)
{
    BEGIN_TEST()
    
    auto A = Tensor<float>({2, 3}, {0, 1, 2, 3, 4, 5});
    int expected[6] = {0, 2, 4, 1, 3, 5};

    int p = 0;
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            EXPECT_NEAR(expected[p++], A.get({i, j}), EPISLON) << "index lookup faulire";
        }
    }

    END_TEST();
}

TEST(Tensor, TensorInitialization_rank3)
{
    BEGIN_TEST()

    std::vector<float> data(24);
    for (int i = 0; i < 24; ++i)
    {
        data[i] = i;
    }

    std::vector<float> expected{0, 8, 16, 4, 12, 20, 1, 9, 17, 5, 13, 21, 2, 10, 18, 6, 14, 22, 3, 11, 19, 7, 15, 23};

    auto A = Tensor<float>({4, 2, 3}, data);

    int p = 0;
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            for (int k = 0; k < 3; ++k)
            {

                float value = A.get({i, j, k});
                EXPECT_NEAR(expected[p++], value, EPISLON) << "index lookup faulire";
            }
        }
    }

    END_TEST();
}

TEST(Tensor, ZeroesInitialization)
{
    BEGIN_TEST()

    Tensor<float> A = Tensor<float>::ZEROES({2, 3});

    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            EXPECT_NEAR(0, A.get({i, j}), EPISLON) << "initialization failure";
        }
    }

    END_TEST();
}

TEST(Tensor, OnesInitialization)
{
    BEGIN_TEST()

    Tensor<float> A = Tensor<float>::ONES({2, 3});

    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            EXPECT_NEAR(1, A.get({i, j}), EPISLON) << "initialization failure";
        }
    }

    END_TEST();
}

TEST(Tensor, RandomInitialization)
{
    BEGIN_TEST()

    std::random_device rnd;
    std::mt19937 mersenne{rnd()};
    std::uniform_real_distribution<float> distribution{-0.05, 0.05};

    auto random_generator = [&mersenne, &distribution]()
    {
        return distribution(mersenne);
    };

    Tensor<float> A = Tensor<float>::RANDOM({2, 3}, random_generator);

    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            EXPECT_NEAR(0, A.get({i, j}), 0.05) << "initialization failure";
        }
    }

    END_TEST();
}

TEST(Tensor, Tensor_set_rank2)
{
    BEGIN_TEST()
    
    auto A = Tensor<float>({2, 3}, {0, 1, 2, 3, 4, 5});
    int expected[6] = {0, 2, -4, -1, 3, 5};

    A.set({1, 0}, -1);
    A.set({0, 2}, -4);

    int p = 0;
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            EXPECT_NEAR(expected[p++], A.get({i, j}), EPISLON) << "index lookup faulire";
        }
    }

    END_TEST();
}

TEST(Tensor, Tensor_set_rank3)
{
    BEGIN_TEST()

    std::vector<float> data(24);
    for (int i = 0; i < 24; ++i)
    {
        data[i] = i;
    }

    std::vector<float> expected{0, 8, 16, 4, 12, -20, 1, 9, 17, 5, 13, 21, 2, -10, 18, 6, 14, 22, 3, 11, 19, -7, 15, 23};

    auto A = Tensor<float>({4, 2, 3}, data);

    A.set({0, 1, 2}, -20);
    A.set({2, 0, 1}, -10);
    A.set({3, 1, 0}, -7);

    int p = 0;
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            for (int k = 0; k < 3; ++k)
            {

                float value = A.get({i, j, k});
                EXPECT_NEAR(expected[p++], value, EPISLON) << "index lookup faulire";
            }
        }
    }

    END_TEST();
}

TEST(Tensor, artihmetic_ops) {
    BEGIN_TEST()

    Tensor<float> A = Tensor<float>::ONES({2, 3});

    Tensor<float> B = Tensor<float>::ONES({2, 3});

    Tensor<float> C = B * 2.f;

    Tensor<float> D = A + C;

    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            EXPECT_NEAR(3.f, D.get({i, j}), EPISLON) << "arithmetic failure";
        }
    }

    END_TEST();
}

TEST(Tensor, artihmetic_ops2) {
    BEGIN_TEST()

    Tensor<float> A = Tensor<float>::ONES({2, 3});

    Tensor<float> B = Tensor<float>::ONES({2, 3});

    Tensor<float> C = B * 2.f;

    Tensor<float> D = A * 3.f;

    Tensor<float> E = C * D;

    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            EXPECT_NEAR(6.f, E.get({i, j}), EPISLON) << "arithmetic failure";
        }
    }

    END_TEST();
}

TEST(Tensor, artihmetic_ops3) {
    BEGIN_TEST()

    Tensor<float> A = Tensor<float>::ONES({2, 3});

    Tensor<float> B = Tensor<float>::ONES({2, 3});

    Tensor<float> C = B * 5.f;

    Tensor<float> D = A * 10.f;

    Tensor<float> E = C / D;

    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            EXPECT_NEAR(.5f, E.get({i, j}), EPISLON) << "arithmetic failure";
        }
    }

    END_TEST();
}

TEST(Tensor, artihmetic_ops4) {
    BEGIN_TEST()

    Tensor<float> A = Tensor<float>::ONES({2, 3});

    Tensor<float> B = Tensor<float>::ONES({2, 3});

    Tensor<float> C = B * 2.f;

    Tensor<float> D = A * 5.f;

    Tensor<float> E = C - D;

    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            EXPECT_NEAR(-3.f, E.get({i, j}), EPISLON) << "arithmetic failure";
        }
    }

    END_TEST();
}

TEST(Tensor, unary_apply_lambda) {
    BEGIN_TEST()

    Tensor<float> A = Tensor<float>::ONES({2, 3});

    const auto lambda = [](float val) {
        return val * 100.f;
    };

    Tensor<float> D = A.unary_apply(lambda);

    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            EXPECT_NEAR(100.f, D.get({i, j}), EPISLON) << "unary apply failure";
        }
    }

    END_TEST();
}

TEST(Tensor, binary_apply_lambda) {
    BEGIN_TEST()

    Tensor<float> A = Tensor<float>::ONES({2, 3});
    Tensor<float> B = Tensor<float>::ONES({2, 3});
    Tensor<float> C = A * 2.f;
    Tensor<float> D = B * 3.f;

    const auto lambda = [](float a, float b) {
        return a * b;
    };

    Tensor<float> E = C.binary_apply(D, lambda);

    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            EXPECT_NEAR(6.f, E.get({i, j}), EPISLON) << "binary apply failure";
        }
    }

    END_TEST();
}
