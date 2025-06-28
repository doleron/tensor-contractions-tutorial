#include <iostream>
#include <vector>
#include <numeric>

int main(int, char **)
{

    std::vector<int> dimensions {6,7,8,9};
    std::vector<int> prefix (dimensions.size(), 1);

    for (int j = dimensions.size() - 2; j >= 0; j--) {
        prefix[j] = dimensions[j + 1] * prefix[j + 1];
    }
    // prefix is {504, 72, 9, 1}

    std::vector<int> indices {3, 1, 4, 1};

    int pos = std::inner_product(indices.begin(), indices.end(), prefix.begin(), 0);

    std::cout << pos << "\n"; // prints 1621

    return 0;
}