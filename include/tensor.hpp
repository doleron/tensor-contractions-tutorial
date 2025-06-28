#ifndef __TENSOR__
#define __TENSOR__

#include <vector>
#include <random>
#include <algorithm>
#include <exception>
#include <limits>
#include <functional>
#include <climits>
#include <unordered_set>
#include <execution>

template <typename FP_TYPE>
class Tensor
{

public:
    using DIMS = std::vector<int>;
    using INDEX_PAIRS = std::vector<std::pair<int, int>>;
    using STORAGE = std::vector<FP_TYPE>;

    constexpr static int MAX_RANK = 10;

    Tensor() {}

    Tensor(const DIMS &dimensions, const STORAGE &data = STORAGE(0))
    {
        this->init(dimensions);
        if (!data.empty())
        {
            if (data.size() != this->storage.capacity())
            {
                throw std::invalid_argument("data size does not match provided dimensions");
            }
            this->storage = data;
        }
    }

    virtual ~Tensor() {}

    template <typename RND_GENERATOR>
    Tensor<FP_TYPE> static RANDOM(const DIMS &dims, RND_GENERATOR &random_generator)
    {
        Tensor<FP_TYPE> result(dims);
        result.storage.resize(result.storage.capacity());
        std::generate(result.storage.begin(), result.storage.end(), random_generator);
        return result;
    }

    Tensor<FP_TYPE> static ZEROES(const DIMS &dims)
    {
        Tensor<FP_TYPE> result(dims);
        result.storage.resize(result.storage.capacity(), FP_TYPE(0));
        return result;
    }

    Tensor<FP_TYPE> static ONES(const DIMS &dims)
    {
        Tensor<FP_TYPE> result(dims);
        result.storage.resize(result.storage.capacity(), FP_TYPE(1));
        return result;
    }

    class INDEX_GENERATOR
    {

    public:
        INDEX_GENERATOR(const DIMS &dims)
        {
            const int DIM_SIZE = dims.size();
            this->divisors.resize(DIM_SIZE, 1);
            this->length = 1;
            for (int j = DIM_SIZE - 1; j >= 0; j--)
            {
                this->divisors[j] = this->length;
                this->length *= dims[j];
            }
        }
        virtual ~INDEX_GENERATOR() {}

        DIMS operator()(const int pos) const
        {
            const int DIM_SIZE = this->divisors.size();
            DIMS result(DIM_SIZE, 0);
            int residual = pos;
            int acc;
            int index;
            for (int j = 0; j < DIM_SIZE; j++)
            {
                acc = this->divisors[j];
                index = residual / acc;
                result[j] = index;
                residual -= index * acc;
            }
            return result;
        }

        int get_length() const
        {
            return this->length;
        }

    private:
        DIMS divisors;
        int length;
    };

    FP_TYPE get(const std::vector<int> &indices) const
    {
        int pos = translate_to_col_major(indices);
        return this->storage[pos];
    }

    void set(const std::vector<int> &indices, FP_TYPE value)
    {
        int pos = translate_to_col_major(indices);
        this->storage[pos] = value;
    }

    Tensor<FP_TYPE> operator+(const Tensor &other) const
    {

        if (!std::equal(this->dimensions.begin(), this->dimensions.end(), other.dimensions.begin()))
        {
            throw std::invalid_argument("You can't sum tensors with different dimensions");
        }

        Tensor<FP_TYPE> result = Tensor<FP_TYPE>::ZEROES(this->dimensions);
        std::transform(this->storage.begin(), this->storage.end(), other.storage.begin(), result.storage.begin(), std::plus<FP_TYPE>());

        return result;
    }

    Tensor<FP_TYPE> operator-(const Tensor<FP_TYPE> &other) const
    {

        if (!std::equal(this->dimensions.begin(), this->dimensions.end(), other.dimensions.begin()))
        {
            throw std::invalid_argument("You can't subtract tensors with different dimensions");
        }

        Tensor<FP_TYPE> result = Tensor<FP_TYPE>::ZEROES(this->dimensions);
        std::transform(this->storage.begin(), this->storage.end(), other.storage.begin(), result.storage.begin(), std::minus<FP_TYPE>());

        return result;
    }

    /**
     * coeff wise product
     */
    Tensor<FP_TYPE> operator*(const Tensor<FP_TYPE> &other) const
    {

        if (!std::equal(this->dimensions.begin(), this->dimensions.end(), other.dimensions.begin()))
        {
            throw std::invalid_argument("You can't multiply tensors with different dimensions");
        }

        Tensor<FP_TYPE> result = Tensor<FP_TYPE>::ZEROES(this->dimensions);
        std::transform(this->storage.begin(), this->storage.end(), other.storage.begin(), result.storage.begin(), std::multiplies<FP_TYPE>());

        return result;
    }

    /**
     * product by scalar
     */
    Tensor<FP_TYPE> operator*(const FP_TYPE scalar) const
    {

        const auto lambda = [&scalar](FP_TYPE value)
        {
            return value * scalar;
        };

        Tensor<FP_TYPE> result = Tensor<FP_TYPE>::ZEROES(this->dimensions);
        std::transform(this->storage.begin(), this->storage.end(), result.storage.begin(), lambda);

        return result;
    }

    Tensor<FP_TYPE> operator/(const Tensor<FP_TYPE> &other) const
    {

        if (!std::equal(this->dimensions.begin(), this->dimensions.end(), other.dimensions.begin()))
        {
            throw std::invalid_argument("You can't divide tensors with different dimensions");
        }

        Tensor<FP_TYPE> result = Tensor<FP_TYPE>::ZEROES(this->dimensions);
        std::transform(this->storage.begin(), this->storage.end(), other.storage.begin(), result.storage.begin(), std::divides<FP_TYPE>());

        return result;
    }

    Tensor<FP_TYPE> unary_apply(const std::function<FP_TYPE(FP_TYPE)> &function) const
    {

        Tensor<FP_TYPE> result = Tensor<FP_TYPE>::ZEROES(this->dimensions);
        std::transform(this->storage.begin(), this->storage.end(), result.storage.begin(), function);

        return result;
    }

    Tensor<FP_TYPE> binary_apply(const Tensor<FP_TYPE> &other, const std::function<FP_TYPE(FP_TYPE, FP_TYPE)> &function) const
    {

        Tensor<FP_TYPE> result = Tensor<FP_TYPE>::ZEROES(this->dimensions);
        std::transform(this->storage.begin(), this->storage.end(), other.storage.begin(), result.storage.begin(), function);

        return result;
    }

    Tensor<FP_TYPE> contraction(const Tensor<FP_TYPE> &other, const INDEX_PAIRS &pairs) const
    {

        const int this_size = this->dimensions.size();
        const int other_size = other.dimensions.size();
        const int pairs_size = pairs.size();
        const int dim_size = this_size + other_size - 2 * pairs_size;

        if (dim_size < 0)
        {
            throw std::invalid_argument("illegal contraction pairs");
        }

        if (!dim_size)
        {
            throw std::invalid_argument("0-size contraction is not supported now");
        }

        DIMS this_pre(dim_size + pairs_size, 0);
        DIMS other_pre(dim_size + pairs_size, 0);

        DIMS intersection_dims(pairs_size, 0);

        std::unordered_set<int> this_pairs_set;
        std::unordered_set<int> other_pairs_set;
        for (int idx = 0; idx < pairs_size; ++idx)
        {
            const auto &pair = pairs[idx];

            if (pair.first < 0 || pair.first >= this->dimensions.size() || this->dimensions[pair.first] < 0)
            {
                throw std::invalid_argument("illegal contraction pairs");
            }
            if (pair.second < 0 || pair.second >= other.dimensions.size() || other.dimensions[pair.second] < 0)
            {
                throw std::invalid_argument("illegal contraction pairs");
            }
            if (this->dimensions[pair.first] != other.dimensions[pair.second])
            {
                throw std::invalid_argument("pairs dimensions do not match");
            }

            this_pairs_set.insert(pair.first);
            other_pairs_set.insert(pair.second);

            this_pre[idx + dim_size] = this->index_lookup_prefix[pair.first];
            other_pre[idx + dim_size] = other.index_lookup_prefix[pair.second];

            intersection_dims[idx] = this->dimensions[pair.first];
        }

        DIMS result_dimensions;
        result_dimensions.reserve(dim_size);

        int dest_pointer = 0;
        for (int i = 0; i < this_size; ++i)
        {
            if (this_pairs_set.find(i) == this_pairs_set.end())
            {
                int dim = this->dimensions[i];
                result_dimensions.push_back(dim);
                this_pre[dest_pointer] = this->index_lookup_prefix[i];
                dest_pointer++;
            }
        }

        for (int i = 0; i < other_size; ++i)
        {
            if (other_pairs_set.find(i) == other_pairs_set.end())
            {
                int dim = other.dimensions[i];
                result_dimensions.push_back(dim);
                other_pre[dest_pointer] = other.index_lookup_prefix[i];
                dest_pointer++;
            }
        }

        Tensor<FP_TYPE> result = ZEROES(result_dimensions);

        auto index_generator = INDEX_GENERATOR(result_dimensions);
        auto intersection_generator = INDEX_GENERATOR(intersection_dims);

        const int length = index_generator.get_length();
        const int intersection_length = intersection_generator.get_length();

        // precaching intersection offsets
        std::vector<int> position_this_cache(intersection_length, 0);
        std::vector<int> position_other_cache(intersection_length, 0);
        for (int j = 0; j < intersection_length; ++j)
        {
            DIMS intersection = intersection_generator(j);
            int position_this = 0;
            int position_other = 0;
            for (int idx = 0; idx < pairs_size; ++idx)
            {
                position_this += intersection[idx] * this_pre[idx + dim_size];
                position_other += intersection[idx] * other_pre[idx + dim_size];
            }
            position_this_cache[j] = position_this;
            position_other_cache[j] = position_other;
        }

        for (int i = 0; i < length; ++i)
        {

            DIMS indices = index_generator(i);

            int pos_this_before = 0;
            int pos_other_before = 0;
            for (int idx = 0; idx < dim_size; ++idx)
            {
                pos_this_before += indices[idx] * this_pre[idx];
                pos_other_before += indices[idx] * other_pre[idx];
            }

            FP_TYPE acc = FP_TYPE(0);
            for (int j = 0; j < intersection_length; ++j)
            {

                int pos_this = pos_this_before + position_this_cache[j];
                int pos_other = pos_other_before + position_other_cache[j];
#ifdef _DEBUG
                if (pos_this < 0 || pos_this >= this->storage.size())
                {
                    throw std::out_of_range("Trying to access illegal self storage position at " + std::to_string(pos_this));
                }

                if (pos_other < 0 || pos_other >= other.storage.size())
                {
                    throw std::out_of_range("Trying to access illegal other storage position at " + std::to_string(pos_other));
                }
#endif

                FP_TYPE this_coeff = this->storage[pos_this];
                FP_TYPE other_coeff = other.storage[pos_other];
                acc += this_coeff * other_coeff;
            }
            result.set(indices, acc);
        }

        return result;
    }

    Tensor<FP_TYPE> contraction_multithread(const Tensor<FP_TYPE> &other, const INDEX_PAIRS &pairs) const
    {

        const int this_size = this->dimensions.size();
        const int other_size = other.dimensions.size();
        const int pairs_size = pairs.size();
        const int dim_size = this_size + other_size - 2 * pairs_size;

        if (dim_size < 0)
        {
            throw std::invalid_argument("illegal contraction pairs");
        }

        if (!dim_size)
        {
            throw std::invalid_argument("0-size contraction is not supported now");
        }

        DIMS this_pre(dim_size + pairs_size, 0);
        DIMS other_pre(dim_size + pairs_size, 0);

        DIMS intersection_dims(pairs_size, 0);

        std::unordered_set<int> this_pairs_set;
        std::unordered_set<int> other_pairs_set;
        for (int idx = 0; idx < pairs_size; ++idx)
        {
            const auto &pair = pairs[idx];

            if (pair.first < 0 || pair.first >= this->dimensions.size() || this->dimensions[pair.first] < 0)
            {
                throw std::invalid_argument("illegal contraction pairs");
            }
            if (pair.second < 0 || pair.second >= other.dimensions.size() || other.dimensions[pair.second] < 0)
            {
                throw std::invalid_argument("illegal contraction pairs");
            }
            if (this->dimensions[pair.first] != other.dimensions[pair.second])
            {
                throw std::invalid_argument("pairs dimensions do not match");
            }

            this_pairs_set.insert(pair.first);
            other_pairs_set.insert(pair.second);

            this_pre[idx + dim_size] = this->index_lookup_prefix[pair.first];
            other_pre[idx + dim_size] = other.index_lookup_prefix[pair.second];

            intersection_dims[idx] = this->dimensions[pair.first];
        }

        DIMS result_dimensions;
        result_dimensions.reserve(dim_size);

        int dest_pointer = 0;
        for (int i = 0; i < this_size; ++i)
        {
            if (this_pairs_set.find(i) == this_pairs_set.end())
            {
                int dim = this->dimensions[i];
                result_dimensions.push_back(dim);
                this_pre[dest_pointer] = this->index_lookup_prefix[i];
                dest_pointer++;
            }
        }

        for (int i = 0; i < other_size; ++i)
        {
            if (other_pairs_set.find(i) == other_pairs_set.end())
            {
                int dim = other.dimensions[i];
                result_dimensions.push_back(dim);
                other_pre[dest_pointer] = other.index_lookup_prefix[i];
                dest_pointer++;
            }
        }

        Tensor<FP_TYPE> result = ZEROES(result_dimensions);

        auto index_generator = INDEX_GENERATOR(result_dimensions);
        auto intersection_generator = INDEX_GENERATOR(intersection_dims);

        const int length = index_generator.get_length();
        const int intersection_length = intersection_generator.get_length();

        // precaching intersection offsets
        std::vector<int> position_this_cache(intersection_length, 0);
        std::vector<int> position_other_cache(intersection_length, 0);
        for (int j = 0; j < intersection_length; ++j)
        {
            DIMS intersection = intersection_generator(j);
            int position_this = 0;
            int position_other = 0;
            for (int idx = 0; idx < pairs_size; ++idx)
            {
                position_this += intersection[idx] * this_pre[idx + dim_size];
                position_other += intersection[idx] * other_pre[idx + dim_size];
            }
            position_this_cache[j] = position_this;
            position_other_cache[j] = position_other;
        }

        const auto compute = [&](int i)
        {
            DIMS indices = index_generator(i);

            int pos_this_before = 0;
            int pos_other_before = 0;
            for (int idx = 0; idx < dim_size; ++idx)
            {
                pos_this_before += indices[idx] * this_pre[idx];
                pos_other_before += indices[idx] * other_pre[idx];
            }

            FP_TYPE acc = FP_TYPE(0);
            for (int j = 0; j < intersection_length; ++j)
            {

                int pos_this = pos_this_before + position_this_cache[j];
                int pos_other = pos_other_before + position_other_cache[j];
#ifdef _DEBUG
                if (pos_this < 0 || pos_this >= this->storage.size())
                {
                    throw std::out_of_range("Trying to access illegal self storage position at " + std::to_string(pos_this));
                }

                if (pos_other < 0 || pos_other >= other.storage.size())
                {
                    throw std::out_of_range("Trying to access illegal other storage position at " + std::to_string(pos_other));
                }
#endif

                FP_TYPE this_coeff = this->storage[pos_this];
                FP_TYPE other_coeff = other.storage[pos_other];
                acc += this_coeff * other_coeff;
            }
            result.set(indices, acc);
        };

        std::vector<int> coefficients(length);
        std::iota(coefficients.begin(), coefficients.end(), 0);
        std::for_each(std::execution::par, coefficients.begin(), coefficients.end(), compute);

        return result;
    }

    const DIMS &get_dimensions() const
    {
        return this->dimensions;
    }

    STORAGE &get_storage()
    {
        return this->storage;
    }

private:
    STORAGE storage;
    DIMS dimensions;
    DIMS index_lookup_prefix;

    void init(const DIMS &dims)
    {
        const int rank = dims.size();
        if (rank > MAX_RANK)
        {
            throw std::invalid_argument("This is a toy implementation and supports only 0-5 Rank tensors but you provided: " + std::to_string(rank));
        }

        int size = 1;
        int prefix = 1;
        const int MAX_STORAGE_SIZE = std::vector<std::vector<FP_TYPE>>().max_size();
        for (int dim : dims)
        {
            if (dim < 1)
            {
                throw std::invalid_argument("Dimension must be positive: " + std::to_string(dim));
            }

            if (dim > (MAX_STORAGE_SIZE / size))
            {
                throw std::invalid_argument("Provided dimensions require a storage larger than the infrastructure capacity.");
            }
            else
            {
                size = size * dim;
            }
            this->index_lookup_prefix.push_back(prefix);
            prefix *= dim;
        }

        // rank 0 tensor
        if (this->index_lookup_prefix.empty())
        {
            this->index_lookup_prefix.push_back(1);
        }

        this->dimensions = dims;
        this->storage.reserve(size);
    }

    inline int translate_to_col_major(const std::vector<int> &indices) const
    {
#ifdef _DEBUG
        if (!indices.size())
        {
            throw std::invalid_argument("no index provided");
        }
        if ((this->dimensions.empty() && indices.size() != 1) || (!this->dimensions.empty() && indices.size() != this->dimensions.size()))
        {
            throw std::invalid_argument("provided indices size do not match with tensor dimensions");
        }
#endif

        int pos = std::inner_product(this->index_lookup_prefix.begin(), this->index_lookup_prefix.end(), indices.begin(), 0);

#ifdef _DEBUG
        if (pos < 0 || pos >= this->storage.size())
        {
            throw std::out_of_range("Trying access a position outside the tensor limits");
        }
#endif

        return pos;
    }
};

#endif