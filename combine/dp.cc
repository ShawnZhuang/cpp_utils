#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>
#include <set>
#include <map>
template <typename Tensor, typename TSpec, typename TensorLess, typename TSpecLess>
class DpState
{
private:
    using VSet = std::set<TSpec, TensorLess>;
    std::map<Tensor, VSet, TensorLess> state;
    using TSelf = DpState<Tensor, TSpec, TensorLess, TSpecLess>;

public:
    DpState() {}
    ~DpState() {}

    void insert(Tensor t, TSpec ts)
    {
        if (state.find(t) == state.end())
        {
            VSet vset;
            vset.insert(ts);
            state[t] = vset;
        }
        else
        {
            state[t].insert(ts);
        }
    }

    bool operator==(const TSelf &other) const
    {
        return state == other.state;
    }

    bool operator<(const TSelf &other) const
    {
        return state < other.state;
    }

    size_t hash() const
    {
        size_t seed = 0;
        for (const auto &pair : state)
        {
            seed ^= std::hash<Tensor>()(pair.first) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            for (const auto &elem : pair.second)
            {
                seed ^= std::hash<TSpec>()(elem) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
        }
        return seed;
    }
};

class DpValue
{
public:
    using Ptr = std::shared_ptr<DpValue>;
    DpValue(/* args */) {}
    ~DpValue() {}

private:
    int64_t cost;

    Ptr prev;
};

template <typename citer>
class ScanIter
{
private:
    using Pair = std::pair<citer, citer>;
    using ValueType = typename std::iterator_traits<citer>::value_type;
    std::queue<Pair> q;

public:
    ScanIter(/* args */) {}
    ScanIter(std::vector<Pair> pairs)
    {
        for (const auto &pair : pairs)
        {
            q.push(pair);
        }
    }
    ~ScanIter() {}
    bool operator==(const ScanIter &other) const
    {
        return q == other.q;
    }
    bool operator!=(const ScanIter &other) const
    {
        return q != other.q;
    }
    citer operator->() const
    {
        return q.front().first;
    }
    ValueType operator*() const
    {
        return *(this->operator->());
    }
    ScanIter &operator++()
    {
        auto pair = q.front();
        pair.first++;
        q.pop();
        if (pair.first != pair.second)
        {
            q.push(pair);
        }
        return *this;
    }
    ScanIter &operator++(int)
    {
        return ++(*this);
    }
};