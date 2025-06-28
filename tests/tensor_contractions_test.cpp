#include <random>

#include <gtest/gtest.h>

// for comparing results

#include "tensor.hpp"
#include "basic_tests_defs.hpp"

const static float EPISLON = 1e-7f;

testing::AssertionResult check_INDEX_GENERATOR(const std::vector<int> dims, int from = 0, int to = -1)
{

    std::vector<int> steps(dims.size(), 1);
    const int dim_size = dims.size();
    if (dim_size > 1)
    {
        int prev = dims[dim_size - 1];
        for (int i = dim_size - 2; i >= 0; i--)
        {
            steps[i] *= prev;
            prev *= dims[i];
        }
    }

    auto generator = Tensor<float>::INDEX_GENERATOR(dims);

    for (int i = from; i < to; ++i)
    {
        const auto &indices = generator(i);
        if (dims.size() != indices.size())
        {
            return testing::AssertionFailure() << "From: " << from << " to: " << to << ". Wrong row size: " << indices.size();
        }

        const int position = std::inner_product(indices.begin(), indices.end(), steps.begin(), 0);
        const int expected = i;
        if (expected != position)
        {
            auto result = testing::AssertionFailure() << "From: " << from << " to: " << to << ". Wrong index at i: " << i << " {";
            for (int dim : dims)
            {
                result << dim << ", ";
            }
            return result << "}";
        }
    }

    return testing::AssertionSuccess();
}

Tensor<float> build_tensor(const Tensor<float>::DIMS dims, float from = 1, float step = 1)
{

    auto generator = Tensor<float>::INDEX_GENERATOR(dims);
    const int length = generator.get_length();
    Tensor<float> result(dims, std::vector<float>(length, 0));
    for (int i = 0; i < length; ++i)
    {
        auto indices = generator(i);
        result.set(indices, i * step + from);
    }
    return result;
}

void print_setup(const Tensor<float>::DIMS &dimsA, const Tensor<float>::DIMS &dimsB, const Tensor<float>::INDEX_PAIRS &pairs)
{

    std::cout << "contraction: {";
    for (int i = 0, size = dimsA.size(); i < size; ++i)
    {
        auto elem = dimsA[i];
        std::cout << elem;
        if (i < size - 1) {
            std::cout << ", ";
        } 
    }
    std::cout << "} x {";

    for (int i = 0, size = dimsB.size(); i < size; ++i)
    {
        auto elem = dimsB[i];
        std::cout << elem;
        if (i < size - 1) {
            std::cout << ", ";
        } 
    }
    std::cout << "} by {";

    for (int i = 0, size = pairs.size(); i < size; ++i)
    {
        auto elem = pairs[i];
        std::cout << "{" << elem.first << ", " << elem.second << "}";
        if (i < size - 1) {
            std::cout << ", ";
        } 
    }

    std::cout << "}\n" << std::flush;
}

TEST(Tensor, generate_indices_rank_3)
{
    BEGIN_TEST()
    EXPECT_TRUE(check_INDEX_GENERATOR({17, 30, 10}));
    EXPECT_TRUE(check_INDEX_GENERATOR({35, 22, 73}));

    EXPECT_TRUE(check_INDEX_GENERATOR({3, 1, 3}));
    EXPECT_TRUE(check_INDEX_GENERATOR({4, 2, 3}));
    EXPECT_TRUE(check_INDEX_GENERATOR({5, 2, 3}));
    EXPECT_TRUE(check_INDEX_GENERATOR({4, 5, 3}));
    EXPECT_TRUE(check_INDEX_GENERATOR({4, 2, 5}));
    EXPECT_TRUE(check_INDEX_GENERATOR({2, 2, 2}));
    EXPECT_TRUE(check_INDEX_GENERATOR({1, 1, 1}));
    EXPECT_TRUE(check_INDEX_GENERATOR({1, 1, 100}));
    EXPECT_TRUE(check_INDEX_GENERATOR({1, 100, 1}));
    EXPECT_TRUE(check_INDEX_GENERATOR({100, 1, 1}));
    EXPECT_TRUE(check_INDEX_GENERATOR({100, 1, 100}));

    END_TEST();
}

TEST(Tensor, generate_indices_rank_2_random)
{
    BEGIN_TEST()

    std::vector<int> dims(2);

    std::random_device rnd;
    std::mt19937 mersenne{rnd()};
    std::uniform_int_distribution<int> distribution{0, 32};

    for (int i = 0; i < 100; ++i)
    {
        for (int j = 0; j < dims.size(); ++j)
        {
            dims[j] = 1 + distribution(mersenne);
        }
        EXPECT_TRUE(check_INDEX_GENERATOR(dims));
    }

    END_TEST();
}

TEST(Tensor, generate_indices_rank_3_random)
{
    BEGIN_TEST()

    std::vector<int> dims(3);

    std::random_device rnd;
    std::mt19937 mersenne{rnd()};
    std::uniform_int_distribution<int> distribution{0, 32};

    for (int i = 0; i < 100; ++i)
    {
        for (int j = 0; j < dims.size(); ++j)
        {
            dims[j] = 1 + distribution(mersenne);
        }
        EXPECT_TRUE(check_INDEX_GENERATOR(dims));
    }

    END_TEST();
}

TEST(Tensor, generate_indices_rank_4_random)
{
    BEGIN_TEST()

    std::vector<int> dims(4);

    std::random_device rnd;
    std::mt19937 mersenne{rnd()};
    std::uniform_int_distribution<int> distribution{0, 32};

    for (int i = 0; i < 100; ++i)
    {
        for (int j = 0; j < dims.size(); ++j)
        {
            dims[j] = 1 + distribution(mersenne);
        }
        EXPECT_TRUE(check_INDEX_GENERATOR(dims));
    }

    END_TEST();
}

TEST(Tensor, generate_indices_rank_3_random_frag)
{
    BEGIN_TEST()

    std::vector<int> dims(3);

    std::random_device rnd;
    std::mt19937 mersenne{rnd()};
    std::uniform_int_distribution<int> distribution{0, 32};

    for (int i = 0; i < 100; ++i)
    {
        int size = 1;
        for (int j = 0; j < dims.size(); ++j)
        {
            dims[j] = 1 + distribution(mersenne);
            size *= dims[j];
        }

        std::uniform_int_distribution<int> from_to_distribution{0, size};
        int from = from_to_distribution(mersenne);
        int to = from_to_distribution(mersenne);

        EXPECT_TRUE(check_INDEX_GENERATOR(dims, from, to));
    }

    END_TEST();
}

TEST(Tensor, generate_indices_rank_5_random)
{
    BEGIN_TEST()

    std::vector<int> dims(5);

    std::random_device rnd;
    std::mt19937 mersenne{rnd()};
    std::uniform_int_distribution<int> distribution{0, 24};

    for (int i = 0; i < 100; ++i)
    {
        for (int j = 0; j < dims.size(); ++j)
        {
            dims[j] = 1 + distribution(mersenne);
        }
        EXPECT_TRUE(check_INDEX_GENERATOR(dims));
    }

    END_TEST();
}

TEST(Tensor, generate_indices_rank_3_big)
{
    BEGIN_TEST()

    std::vector<int> dims{512, 512, 256};
    EXPECT_TRUE(check_INDEX_GENERATOR(dims));

    END_TEST();
}

TEST(Tensor, basic_2d_contractions)
{
    BEGIN_TEST()

    Tensor<float> A, B, C;
    Tensor<float>::DIMS dimensions;

    /*
        | 1 2 3 |   | 1 2 |    | 22 28 |
        | 4 5 6 | x | 3 4 |  = | 49 64 |
                    | 5 6 |
    */

    A = Tensor<float>({2, 3}, {1, 4, 2, 5, 3, 6});
    B = Tensor<float>({3, 2}, {1, 3, 5, 2, 4, 6});

    C = A.contraction(B, {{1, 0}});

    dimensions = C.get_dimensions();

    ASSERT_EQ(2, dimensions.size()) << "wrong rank";

    ASSERT_EQ(2, dimensions[0]) << "wrong rank";

    ASSERT_EQ(2, dimensions[1]) << "wrong rank";

    ASSERT_NEAR(C.get({0, 0}), 22.f, EPISLON);
    ASSERT_NEAR(C.get({0, 1}), 28.f, EPISLON);
    ASSERT_NEAR(C.get({1, 0}), 49.f, EPISLON);
    ASSERT_NEAR(C.get({1, 1}), 64.f, EPISLON);

    /*
        | 1 2 3 |   | 1 2 |    | 22 28 |
        | 4 5 6 | x | 3 4 |  = | 49 64 |
        | 7 8 9 |   | 5 6 |    | 76 100|
    */

    A = Tensor<float>({3, 3}, {1, 4, 7, 2, 5, 8, 3, 6, 9});
    B = Tensor<float>({3, 2}, {1, 3, 5, 2, 4, 6});

    C = A.contraction(B, {{1, 0}});

    dimensions = C.get_dimensions();

    ASSERT_EQ(2, dimensions.size()) << "wrong rank";

    ASSERT_EQ(3, dimensions[0]) << "wrong rank";

    ASSERT_EQ(2, dimensions[1]) << "wrong rank";

    ASSERT_NEAR(C.get({0, 0}), 22.f, EPISLON);
    ASSERT_NEAR(C.get({0, 1}), 28.f, EPISLON);
    ASSERT_NEAR(C.get({1, 0}), 49.f, EPISLON);
    ASSERT_NEAR(C.get({1, 1}), 64.f, EPISLON);
    ASSERT_NEAR(C.get({2, 0}), 76.f, EPISLON);
    ASSERT_NEAR(C.get({2, 1}), 100.f, EPISLON);

    /*
        | 1 4 7 |   | 1 2 |    | 22 28 |
        | 2 5 8 | x | 3 4 |  = | 49 64 |
        | 3 6 9 |   | 5 6 |    | 76 100|
    */

    A = Tensor<float>({3, 3}, {1, 4, 7, 2, 5, 8, 3, 6, 9});
    B = Tensor<float>({3, 2}, {1, 3, 5, 2, 4, 6});

    C = A.contraction(B, {{0, 0}});

    dimensions = C.get_dimensions();

    ASSERT_EQ(2, dimensions.size()) << "wrong rank";

    ASSERT_EQ(3, dimensions[0]) << "wrong rank";

    ASSERT_EQ(2, dimensions[1]) << "wrong rank";

    ASSERT_NEAR(C.get({0, 0}), 48.f, EPISLON);
    ASSERT_NEAR(C.get({0, 1}), 60.f, EPISLON);
    ASSERT_NEAR(C.get({1, 0}), 57.f, EPISLON);
    ASSERT_NEAR(C.get({1, 1}), 72.f, EPISLON);
    ASSERT_NEAR(C.get({2, 0}), 66.f, EPISLON);
    ASSERT_NEAR(C.get({2, 1}), 84.f, EPISLON);

    /*
        | 1 2 3 |   | 1 2 3 |    | 17 22 27 |
        | 4 5 6 | x | 4 5 6 |  = | 22 29 36 |
                                 | 27 36 45 |
    */

    A = Tensor<float>({2, 3}, {1, 4, 2, 5, 3, 6});
    B = Tensor<float>({2, 3}, {1, 4, 2, 5, 3, 6});

    C = A.contraction(B, {{0, 0}});

    dimensions = C.get_dimensions();

    ASSERT_EQ(2, dimensions.size()) << "wrong rank";

    ASSERT_EQ(3, dimensions[0]) << "wrong rank";

    ASSERT_EQ(3, dimensions[1]) << "wrong rank";

    ASSERT_NEAR(C.get({0, 0}), 17.f, EPISLON);
    ASSERT_NEAR(C.get({0, 1}), 22.f, EPISLON);
    ASSERT_NEAR(C.get({0, 2}), 27.f, EPISLON);
    ASSERT_NEAR(C.get({1, 0}), 22.f, EPISLON);
    ASSERT_NEAR(C.get({1, 1}), 29.f, EPISLON);
    ASSERT_NEAR(C.get({1, 2}), 36.f, EPISLON);
    ASSERT_NEAR(C.get({2, 0}), 27.f, EPISLON);
    ASSERT_NEAR(C.get({2, 1}), 36.f, EPISLON);
    ASSERT_NEAR(C.get({2, 2}), 45.f, EPISLON);

    /*
        | 1 2 3 |   | 1 2 3 |    | 17 22 27 |
        | 4 5 6 | x | 4 5 6 |  = | 22 29 36 |
                                 | 27 36 45 |
    */

    A = Tensor<float>({2, 3}, {1, 4, 2, 5, 3, 6});
    B = Tensor<float>({2, 3}, {1, 4, 2, 5, 3, 6});

    C = A.contraction(B, {{1, 1}});

    dimensions = C.get_dimensions();

    ASSERT_EQ(2, dimensions.size()) << "wrong rank";

    ASSERT_EQ(2, dimensions[0]) << "wrong rank";

    ASSERT_EQ(2, dimensions[1]) << "wrong rank";

    ASSERT_NEAR(C.get({0, 0}), 14.f, EPISLON);
    ASSERT_NEAR(C.get({0, 1}), 32.f, EPISLON);
    ASSERT_NEAR(C.get({1, 0}), 32.f, EPISLON);
    ASSERT_NEAR(C.get({1, 1}), 77.f, EPISLON);

    A = build_tensor({10, 7});
    B = build_tensor({5, 10});

    C = A.contraction(B, {{0, 1}});

    dimensions = C.get_dimensions();

    ASSERT_EQ(2, dimensions.size()) << "wrong rank";

    ASSERT_EQ(7, dimensions[0]) << "wrong rank";

    ASSERT_EQ(5, dimensions[1]) << "wrong rank";

    auto storage = C.get_storage();
    std::vector<float> expected{2365, 2420, 2475, 2530, 2585, 2640, 2695, 5615, 5770,
                                5925, 6080, 6235, 6390, 6545, 8865, 9120, 9375, 9630, 9885, 10140, 10395, 12115,
                                12470, 12825, 13180, 13535, 13890, 14245, 15365, 15820, 16275, 16730, 17185, 17640, 18095};

    for (int i = 0, limit = 5 * 7; i < limit; ++i)
    {
        ASSERT_NEAR(storage[i], expected[i], EPISLON) << "wrong value at i = " << i;
    }

    A = build_tensor({7, 11});
    B = build_tensor({11, 5});

    C = A.contraction(B, {{1, 0}});

    dimensions = C.get_dimensions();

    ASSERT_EQ(2, dimensions.size()) << "wrong rank";

    ASSERT_EQ(7, dimensions[0]) << "wrong rank";

    ASSERT_EQ(5, dimensions[1]) << "wrong rank";

    storage = C.get_storage();
    expected = std::vector<float>{2266, 5412, 8558, 11704, 14850, 17996, 21142, 2332, 5599, 8866, 12133,
                                  15400, 18667, 21934, 2398, 5786, 9174, 12562, 15950, 19338, 22726, 2464, 5973, 9482, 12991, 16500,
                                  20009, 23518, 2530, 6160, 9790, 13420, 17050, 20680, 24310};

    for (int i = 0, limit = 5 * 7; i < limit; ++i)
    {
        ASSERT_NEAR(storage[i], expected[i], EPISLON) << "wrong value at i = " << i;
    }

    A = Tensor<float>({1, 3}, {1, 5, 3});
    B = Tensor<float>({1, 3}, {2, 10, 1});

    C = A.contraction(B, {{1, 1}});

    dimensions = C.get_dimensions();

    ASSERT_EQ(2, dimensions.size()) << "wrong rank";

    ASSERT_EQ(1, dimensions[0]) << "wrong rank";

    ASSERT_EQ(1, dimensions[1]) << "wrong rank";

    ASSERT_NEAR(C.get({0, 0}), 55.f, EPISLON);

    END_TEST();
}

TEST(Tensor, basic_2d_contractions_2)
{
    BEGIN_TEST()

    std::random_device rnd;
    std::mt19937 mersenne{rnd()};
    std::uniform_real_distribution<float> distribution{-0.05, 0.05};

    auto random_generator = [&mersenne, &distribution]()
    {
        return distribution(mersenne);
    };

    Tensor<float> mat1 = Tensor<float>::RANDOM({2, 3}, random_generator);
    Tensor<float> mat2 = Tensor<float>::RANDOM({2, 3}, random_generator);

    Tensor<float> mat4 = mat1.contraction(mat2, {{0, 0}});

    ASSERT_EQ(2, mat4.get_dimensions().size());
    ASSERT_EQ(mat4.get_dimensions()[0], 3);
    ASSERT_EQ(mat4.get_dimensions()[1], 3);

    ASSERT_NEAR(mat4.get({0, 0}), mat1.get({0, 0}) * mat2.get({0, 0}) + mat1.get({1, 0}) * mat2.get({1, 0}), EPISLON);
    ASSERT_NEAR(mat4.get({0, 1}), mat1.get({0, 0}) * mat2.get({0, 1}) + mat1.get({1, 0}) * mat2.get({1, 1}), EPISLON);
    ASSERT_NEAR(mat4.get({0, 2}), mat1.get({0, 0}) * mat2.get({0, 2}) + mat1.get({1, 0}) * mat2.get({1, 2}), EPISLON);
    ASSERT_NEAR(mat4.get({1, 0}), mat1.get({0, 1}) * mat2.get({0, 0}) + mat1.get({1, 1}) * mat2.get({1, 0}), EPISLON);
    ASSERT_NEAR(mat4.get({1, 1}), mat1.get({0, 1}) * mat2.get({0, 1}) + mat1.get({1, 1}) * mat2.get({1, 1}), EPISLON);
    ASSERT_NEAR(mat4.get({1, 2}), mat1.get({0, 1}) * mat2.get({0, 2}) + mat1.get({1, 1}) * mat2.get({1, 2}), EPISLON);
    ASSERT_NEAR(mat4.get({2, 0}), mat1.get({0, 2}) * mat2.get({0, 0}) + mat1.get({1, 2}) * mat2.get({1, 0}), EPISLON);
    ASSERT_NEAR(mat4.get({2, 1}), mat1.get({0, 2}) * mat2.get({0, 1}) + mat1.get({1, 2}) * mat2.get({1, 1}), EPISLON);
    ASSERT_NEAR(mat4.get({2, 2}), mat1.get({0, 2}) * mat2.get({0, 2}) + mat1.get({1, 2}) * mat2.get({1, 2}), EPISLON);

    auto mat5 = mat1.contraction(mat2, {{1, 1}});

    ASSERT_EQ(2, mat5.get_dimensions().size());
    ASSERT_EQ(mat5.get_dimensions()[0], 2);
    ASSERT_EQ(mat5.get_dimensions()[1], 2);

    ASSERT_NEAR(mat5.get({0, 0}), mat1.get({0, 0}) * mat2.get({0, 0}) + mat1.get({0, 1}) * mat2.get({0, 1}) + mat1.get({0, 2}) * mat2.get({0, 2}), EPISLON);
    ASSERT_NEAR(mat5.get({0, 1}), mat1.get({0, 0}) * mat2.get({1, 0}) + mat1.get({0, 1}) * mat2.get({1, 1}) + mat1.get({0, 2}) * mat2.get({1, 2}), EPISLON);
    ASSERT_NEAR(mat5.get({1, 0}), mat1.get({1, 0}) * mat2.get({0, 0}) + mat1.get({1, 1}) * mat2.get({0, 1}) + mat1.get({1, 2}) * mat2.get({0, 2}), EPISLON);
    ASSERT_NEAR(mat5.get({1, 1}), mat1.get({1, 0}) * mat2.get({1, 0}) + mat1.get({1, 1}) * mat2.get({1, 1}) + mat1.get({1, 2}) * mat2.get({1, 2}), EPISLON);

    Tensor<float> mat3 = Tensor<float>::RANDOM({3, 2}, random_generator);
    Tensor<float> mat6 = mat1.contraction(mat3, {{1, 0}});

    ASSERT_EQ(2, mat6.get_dimensions().size());
    ASSERT_EQ(mat6.get_dimensions()[0], 2);
    ASSERT_EQ(mat6.get_dimensions()[1], 2);

    ASSERT_NEAR(mat6.get({0, 0}), mat1.get({0, 0}) * mat3.get({0, 0}) + mat1.get({0, 1}) * mat3.get({1, 0}) + mat1.get({0, 2}) * mat3.get({2, 0}), EPISLON);
    ASSERT_NEAR(mat6.get({0, 1}), mat1.get({0, 0}) * mat3.get({0, 1}) + mat1.get({0, 1}) * mat3.get({1, 1}) + mat1.get({0, 2}) * mat3.get({2, 1}), EPISLON);
    ASSERT_NEAR(mat6.get({1, 0}), mat1.get({1, 0}) * mat3.get({0, 0}) + mat1.get({1, 1}) * mat3.get({1, 0}) + mat1.get({1, 2}) * mat3.get({2, 0}), EPISLON);
    ASSERT_NEAR(mat6.get({1, 1}), mat1.get({1, 0}) * mat3.get({0, 1}) + mat1.get({1, 1}) * mat3.get({1, 1}) + mat1.get({1, 2}) * mat3.get({2, 1}), EPISLON);

    END_TEST();
}

TEST(Tensor, test_multidims)
{

    std::random_device rnd;
    std::mt19937 mersenne{rnd()};
    std::uniform_real_distribution<float> distribution{-0.05, 0.05};

    auto random_generator = [&mersenne, &distribution]()
    {
        return distribution(mersenne);
    };

    BEGIN_TEST()

    Tensor<float> mat1 = Tensor<float>::RANDOM({2, 2, 2}, random_generator);
    Tensor<float> mat2 = Tensor<float>::RANDOM({2, 2, 2, 2}, random_generator);

    Tensor<float> mat3 = mat1.contraction(mat2, {{1, 2}, {2, 3}});

    ASSERT_EQ(3, mat3.get_dimensions().size());
    ASSERT_EQ(mat3.get_dimensions()[0], 2);
    ASSERT_EQ(mat3.get_dimensions()[1], 2);
    ASSERT_EQ(mat3.get_dimensions()[2], 2);

    ASSERT_NEAR(mat3.get({0, 0, 0}), mat1.get({0, 0, 0}) * mat2.get({0, 0, 0, 0}) + mat1.get({0, 1, 0}) * mat2.get({0, 0, 1, 0}) + mat1.get({0, 0, 1}) * mat2.get({0, 0, 0, 1}) + mat1.get({0, 1, 1}) * mat2.get({0, 0, 1, 1}), EPISLON);
    ASSERT_NEAR(mat3.get({0, 0, 1}), mat1.get({0, 0, 0}) * mat2.get({0, 1, 0, 0}) + mat1.get({0, 1, 0}) * mat2.get({0, 1, 1, 0}) + mat1.get({0, 0, 1}) * mat2.get({0, 1, 0, 1}) + mat1.get({0, 1, 1}) * mat2.get({0, 1, 1, 1}), EPISLON);
    ASSERT_NEAR(mat3.get({0, 1, 0}), mat1.get({0, 0, 0}) * mat2.get({1, 0, 0, 0}) + mat1.get({0, 1, 0}) * mat2.get({1, 0, 1, 0}) + mat1.get({0, 0, 1}) * mat2.get({1, 0, 0, 1}) + mat1.get({0, 1, 1}) * mat2.get({1, 0, 1, 1}), EPISLON);
    ASSERT_NEAR(mat3.get({0, 1, 1}), mat1.get({0, 0, 0}) * mat2.get({1, 1, 0, 0}) + mat1.get({0, 1, 0}) * mat2.get({1, 1, 1, 0}) + mat1.get({0, 0, 1}) * mat2.get({1, 1, 0, 1}) + mat1.get({0, 1, 1}) * mat2.get({1, 1, 1, 1}), EPISLON);
    ASSERT_NEAR(mat3.get({1, 0, 0}), mat1.get({1, 0, 0}) * mat2.get({0, 0, 0, 0}) + mat1.get({1, 1, 0}) * mat2.get({0, 0, 1, 0}) + mat1.get({1, 0, 1}) * mat2.get({0, 0, 0, 1}) + mat1.get({1, 1, 1}) * mat2.get({0, 0, 1, 1}), EPISLON);
    ASSERT_NEAR(mat3.get({1, 0, 1}), mat1.get({1, 0, 0}) * mat2.get({0, 1, 0, 0}) + mat1.get({1, 1, 0}) * mat2.get({0, 1, 1, 0}) + mat1.get({1, 0, 1}) * mat2.get({0, 1, 0, 1}) + mat1.get({1, 1, 1}) * mat2.get({0, 1, 1, 1}), EPISLON);
    ASSERT_NEAR(mat3.get({1, 1, 0}), mat1.get({1, 0, 0}) * mat2.get({1, 0, 0, 0}) + mat1.get({1, 1, 0}) * mat2.get({1, 0, 1, 0}) + mat1.get({1, 0, 1}) * mat2.get({1, 0, 0, 1}) + mat1.get({1, 1, 1}) * mat2.get({1, 0, 1, 1}), EPISLON);
    ASSERT_NEAR(mat3.get({1, 1, 1}), mat1.get({1, 0, 0}) * mat2.get({1, 1, 0, 0}) + mat1.get({1, 1, 0}) * mat2.get({1, 1, 1, 0}) + mat1.get({1, 0, 1}) * mat2.get({1, 1, 0, 1}) + mat1.get({1, 1, 1}) * mat2.get({1, 1, 1, 1}), EPISLON);

    Tensor<float> mat4 = Tensor<float>::RANDOM({2, 2}, random_generator);
    Tensor<float> mat5 = Tensor<float>::RANDOM({2, 2, 2}, random_generator);

    Tensor<float> mat6 = mat4.contraction(mat5, {{0, 1}, {1, 0}});
    ASSERT_EQ(1, mat6.get_dimensions().size());
    ASSERT_EQ(mat6.get_dimensions()[0], 2);

    ASSERT_NEAR(mat6.get({0}), mat4.get({0, 0}) * mat5.get({0, 0, 0}) + mat4.get({1, 0}) * mat5.get({0, 1, 0}) + mat4.get({0, 1}) * mat5.get({1, 0, 0}) + mat4.get({1, 1}) * mat5.get({1, 1, 0}), EPISLON);
    ASSERT_NEAR(mat6.get({1}), mat4.get({0, 0}) * mat5.get({0, 0, 1}) + mat4.get({1, 0}) * mat5.get({0, 1, 1}) + mat4.get({0, 1}) * mat5.get({1, 0, 1}) + mat4.get({1, 1}) * mat5.get({1, 1, 1}), EPISLON);

    END_TEST();
}

TEST(Tensor, test_holes)
{

    std::random_device rnd;
    std::mt19937 mersenne{rnd()};
    std::uniform_real_distribution<float> distribution{-0.05, 0.05};

    auto random_generator = [&mersenne, &distribution]()
    {
        return distribution(mersenne);
    };

    BEGIN_TEST()
    Tensor<float> t1 = Tensor<float>::RANDOM({2, 5, 7, 3}, random_generator);
    Tensor<float> t2 = Tensor<float>::RANDOM({2, 7, 11, 13, 3}, random_generator);

    Tensor<float> result = t1.contraction(t2, {{0, 0}, {3, 4}});

    ASSERT_EQ(result.get_dimensions().size(), 5);

    ASSERT_EQ(result.get_dimensions()[0], 5);
    ASSERT_EQ(result.get_dimensions()[1], 7);
    ASSERT_EQ(result.get_dimensions()[2], 7);
    ASSERT_EQ(result.get_dimensions()[3], 11);
    ASSERT_EQ(result.get_dimensions()[4], 13);

    for (int i = 0; i < 5; ++i)
    {
        for (int j = 0; j < 5; ++j)
        {
            for (int k = 0; k < 5; ++k)
            {
                for (int l = 0; l < 5; ++l)
                {
                    for (int m = 0; m < 5; ++m)
                    {
                        ASSERT_NEAR(result.get({i, j, k, l, m}),
                                    t1.get({0, i, j, 0}) * t2.get({0, k, l, m, 0}) + t1.get({1, i, j, 0}) * t2.get({1, k, l, m, 0}) +
                                        t1.get({0, i, j, 1}) * t2.get({0, k, l, m, 1}) + t1.get({1, i, j, 1}) * t2.get({1, k, l, m, 1}) +
                                        t1.get({0, i, j, 2}) * t2.get({0, k, l, m, 2}) + t1.get({1, i, j, 2}) * t2.get({1, k, l, m, 2}),
                                    EPISLON);
                    }
                }
            }
        }
    }

    END_TEST();
}

TEST(Tensor, test_tensor_product)
{

    std::random_device rnd;
    std::mt19937 mersenne{rnd()};
    std::uniform_real_distribution<float> distribution{-0.05, 0.05};

    auto random_generator = [&mersenne, &distribution]()
    {
        return distribution(mersenne);
    };

    BEGIN_TEST()
    Tensor<float> mat1 = Tensor<float>::RANDOM({2, 3}, random_generator);
    Tensor<float> mat2 = Tensor<float>::RANDOM({4, 1}, random_generator);

    Tensor<float> result = mat1.contraction(mat2, {});

    ASSERT_EQ(result.get_dimensions().size(), 4);

    ASSERT_EQ(result.get_dimensions()[0], 2);
    ASSERT_EQ(result.get_dimensions()[1], 3);
    ASSERT_EQ(result.get_dimensions()[2], 4);
    ASSERT_EQ(result.get_dimensions()[3], 1);

    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            for (int k = 0; k < 4; ++k)
            {
                for (int l = 0; l < 1; ++l)
                {
                    ASSERT_NEAR(result.get({i, j, k, l}), mat1.get({i, j}) * mat2.get({k, l}), EPISLON);
                }
            }
        }
    }

    END_TEST();
}

TEST(Tensor, test_out_of_order_contraction)
{

    std::random_device rnd;
    std::mt19937 mersenne{rnd()};
    std::uniform_real_distribution<float> distribution{-0.05, 0.05};

    auto random_generator = [&mersenne, &distribution]()
    {
        return distribution(mersenne);
    };

    BEGIN_TEST()
    Tensor<float> mat1 = Tensor<float>::RANDOM({2, 2, 2}, random_generator);
    Tensor<float> mat2 = Tensor<float>::RANDOM({2, 2, 2}, random_generator);

    Tensor<float> mat3 = mat1.contraction(mat2, {{2, 0}, {0, 2}});

    ASSERT_NEAR(mat3.get({0, 0}), mat1.get({0, 0, 0}) * mat2.get({0, 0, 0}) + mat1.get({1, 0, 0}) * mat2.get({0, 0, 1}) + mat1.get({0, 0, 1}) * mat2.get({1, 0, 0}) + mat1.get({1, 0, 1}) * mat2.get({1, 0, 1}), EPISLON);
    ASSERT_NEAR(mat3.get({1, 0}), mat1.get({0, 1, 0}) * mat2.get({0, 0, 0}) + mat1.get({1, 1, 0}) * mat2.get({0, 0, 1}) + mat1.get({0, 1, 1}) * mat2.get({1, 0, 0}) + mat1.get({1, 1, 1}) * mat2.get({1, 0, 1}), EPISLON);
    ASSERT_NEAR(mat3.get({0, 1}), mat1.get({0, 0, 0}) * mat2.get({0, 1, 0}) + mat1.get({1, 0, 0}) * mat2.get({0, 1, 1}) + mat1.get({0, 0, 1}) * mat2.get({1, 1, 0}) + mat1.get({1, 0, 1}) * mat2.get({1, 1, 1}), EPISLON);
    ASSERT_NEAR(mat3.get({1, 1}), mat1.get({0, 1, 0}) * mat2.get({0, 1, 0}) + mat1.get({1, 1, 0}) * mat2.get({0, 1, 1}) + mat1.get({0, 1, 1}) * mat2.get({1, 1, 0}) + mat1.get({1, 1, 1}) * mat2.get({1, 1, 1}), EPISLON);

    mat3 = mat1.contraction(mat2, {{0, 2}, {2, 0}});

    ASSERT_NEAR(mat3.get({0, 0}), mat1.get({0, 0, 0}) * mat2.get({0, 0, 0}) + mat1.get({1, 0, 0}) * mat2.get({0, 0, 1}) + mat1.get({0, 0, 1}) * mat2.get({1, 0, 0}) + mat1.get({1, 0, 1}) * mat2.get({1, 0, 1}), EPISLON);
    ASSERT_NEAR(mat3.get({1, 0}), mat1.get({0, 1, 0}) * mat2.get({0, 0, 0}) + mat1.get({1, 1, 0}) * mat2.get({0, 0, 1}) + mat1.get({0, 1, 1}) * mat2.get({1, 0, 0}) + mat1.get({1, 1, 1}) * mat2.get({1, 0, 1}), EPISLON);
    ASSERT_NEAR(mat3.get({0, 1}), mat1.get({0, 0, 0}) * mat2.get({0, 1, 0}) + mat1.get({1, 0, 0}) * mat2.get({0, 1, 1}) + mat1.get({0, 0, 1}) * mat2.get({1, 1, 0}) + mat1.get({1, 0, 1}) * mat2.get({1, 1, 1}), EPISLON);
    ASSERT_NEAR(mat3.get({1, 1}), mat1.get({0, 1, 0}) * mat2.get({0, 1, 0}) + mat1.get({1, 1, 0}) * mat2.get({0, 1, 1}) + mat1.get({0, 1, 1}) * mat2.get({1, 1, 0}) + mat1.get({1, 1, 1}) * mat2.get({1, 1, 1}), EPISLON);

    END_TEST();
}

bool dimension_is_valid(const Tensor<float>::DIMS &dims)
{
    bool result = true;
    int acc = 1;
    const int MAX_STORAGE_SIZE = std::vector<std::vector<float>>().max_size();
    for (int dim : dims)
    {
        if (dim > MAX_STORAGE_SIZE / dim)
        {
            result = false;
            break;
        }
        acc *= dim;
    }
    return result;
}

TEST(Tensor, contraction_dimensions_rand)
{

    std::random_device rnd;
    std::mt19937 mersenne{rnd()};
    std::uniform_real_distribution<float> distribution{-0.05, 0.05};
    std::uniform_int_distribution<int> rank_distribution{1, Tensor<float>::MAX_RANK};
    std::uniform_int_distribution<int> dim_distribution{1, 8};

    auto random_generator = [&mersenne, &distribution]()
    {
        return distribution(mersenne);
    };

    Tensor<float>::DIMS A_dims;
    Tensor<float>::DIMS B_dims;
    Tensor<float>::INDEX_PAIRS pairs;

    try
    {
        for (int epoch = 0; epoch < 100; ++epoch)
        {
            int rank_a = rank_distribution(mersenne);
            int rank_b = rank_distribution(mersenne);
            int min_rank = std::min(rank_a, rank_b);

            int min_pairs_size = std::max(0, (rank_a + rank_b - Tensor<float>::MAX_RANK + 1) / 2);

            int pairs_size = min_pairs_size;
            if (min_pairs_size < min_rank - 1)
            {
                std::uniform_int_distribution<int> pairs_distribution(min_pairs_size, min_rank - 1);
                pairs_size = pairs_distribution(mersenne);
            }
            int test_tensor_rank = rank_a + rank_b - 2 * pairs_size;

            if (test_tensor_rank > Tensor<float>::MAX_RANK)
            {
                FAIL() << "Can't run test with tensor rank: " << test_tensor_rank << " gt: " << Tensor<float>::MAX_RANK << " a: " << rank_a << " b: " << rank_b << " pairs " << pairs_size;
            }

            A_dims = Tensor<float>::DIMS(rank_a, 0);
            B_dims = Tensor<float>::DIMS(rank_b, 0);

            pairs = Tensor<float>::INDEX_PAIRS(pairs_size);

            std::uniform_int_distribution<int> a_distribution_index(0, rank_a - 1);
            std::uniform_int_distribution<int> b_distribution_index(0, rank_b - 1);
            int i = 0;
            int watch_dog = 0;
            const int watch_dog_limit = pairs_size * 100;
            while (i < pairs_size)
            {
                if (watch_dog >= watch_dog_limit)
                {
                    std::cout << "\nCannot fill contraction pairs. Skipping\n";
                    break;
                }

                int a_index = a_distribution_index(mersenne);
                int b_index = b_distribution_index(mersenne);
                if (!A_dims[a_index] && !B_dims[b_index])
                {
                    int dim = dim_distribution(mersenne);
                    A_dims[a_index] = dim;
                    B_dims[b_index] = dim;
                    pairs[i] = std::make_pair(a_index, b_index);
                    i++;
                }
                watch_dog++;
            }
            if (pairs_size > 0 && watch_dog >= watch_dog_limit) {
                continue;
            }

            Tensor<float>::DIMS expected_dimensions;
            for (int j = 0; j < rank_a; ++j)
            {
                if (!A_dims[j])
                {
                    A_dims[j] = dim_distribution(mersenne);
                    expected_dimensions.push_back(A_dims[j]);
                }
            }

            for (int j = 0; j < rank_b; ++j)
            {
                if (!B_dims[j])
                {
                    B_dims[j] = dim_distribution(mersenne);
                    expected_dimensions.push_back(B_dims[j]);
                }
            }

            if (!dimension_is_valid(A_dims) || !dimension_is_valid(B_dims))
            {
                std::cout << "configuration exceeds integer capacity. Skiping\n";
                print_setup(A_dims, B_dims, pairs);
                continue;
            }

            Tensor<float> A = build_tensor(A_dims);
            Tensor<float> B = build_tensor(B_dims);
            std::cout << "epoch: " << epoch << " ";
            print_setup(A_dims, B_dims, pairs);
            Tensor<float> C = A.contraction(B, pairs);

            auto dimensions = C.get_dimensions();

            const int expected_size = expected_dimensions.size();

            ASSERT_EQ(expected_size, dimensions.size()) << "wrong rank";

            for (i = 0; i < expected_size; ++i)
            {
                ASSERT_EQ(expected_dimensions[i], dimensions[i]) << "wrong rank at i = " << i;
            }
        }
    }
    catch (const std::exception &ex)
    {
        print_setup(A_dims, B_dims, pairs);
        FAIL() << "Caught exception: " << ex.what();
    }
    catch (...)
    {
        print_setup(A_dims, B_dims, pairs);
        FAIL() << "Caught unknown exception.";
    }
}