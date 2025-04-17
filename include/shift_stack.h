#ifndef SHIFT_STACK_H
#define SHIFT_STACK_H

#include <deque>
#include <functional>

template <typename T, typename TEquals>
class ShiftStack
{
    std::deque<T> stack;

public:
    ShiftStack() = default;
    ~ShiftStack() = default;

    bool SetTop(std::function<bool(const T &)> fcheck)
    {
        auto depth = stack.size();
        for (int i = 0; i < depth; ++i)
        {
            if (fcheck(stack.back()))
            {
                return true;
            }
            T t = stack.back();
            stack.pop_back();
            stack.push_front(t);
        }
        return false;
    }

    bool SetTop(const T &e)
    {
        TEquals t_equals;
        return SetTop([&](const T &n)
                      { return t_equals(e, n); });
    }

    void push(const T &value)
    {
        // push back
        stack.push_back(value);
    }

    void pop()
    {
        // pop back
        stack.pop_back();
    }
};

#endif // SHIFT_STACK_H