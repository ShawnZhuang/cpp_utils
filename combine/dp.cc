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
    Equals

};