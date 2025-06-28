#include <iostream>
#include <chrono>
#include <random>

#include "tensor.hpp"

int main(int, char **)
{
    std::random_device rnd;
    std::mt19937 mersenne{rnd()};
    std::uniform_real_distribution<float> distribution{-0.05, 0.05};
    auto random_generator = [&mersenne, &distribution]()
    {
        return distribution(mersenne);
    };

    Tensor<float> X = Tensor<float>::RANDOM({31, 4, 32, 3, 28}, random_generator);
    Tensor<float> M = Tensor<float>::RANDOM({{28, 31, 28, 3}}, random_generator);
    Tensor<float>::INDEX_PAIRS pairs{{3, 3}, {0, 1}, {4, 2}};
    Tensor<float> Y;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 10; ++i)
    {
        Y = X.contraction(M, pairs);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = end_time - start_time;
    long long milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    std::cout << "Elapsed time: " << milliseconds << " milliseconds" << "\n\n";

    auto dimensions = Y.get_dimensions();

    std::cout << "X {";

    for (auto elem : X.get_dimensions())
    {
        std::cout << elem << ", ";
    }

    std::cout << "}\nY {";

    for (auto elem : M.get_dimensions())
    {
        std::cout << elem << ", ";
    }

    std::cout << "}\nPairs {";

    for (auto elem : pairs)
    {

        std::cout << "{" << elem.first << ", " << elem.second << "}, ";
    }

    std::cout << "}\n\nsuccess\n";

    return 0;
}