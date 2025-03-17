#include <unordered_map>
#include <tuple>
#include <functional>
/**
 * FuncCache is a template class that caches the results of a function call
 * to avoid redundant calculations. It uses a hash map to store the results
 * of function calls with specific arguments.
 *
 * @tparam Result The return type of the function to be cached.
 * @tparam Args The argument types of the function to be cached.
 */

template<typename Result, typename... Args>
class FuncCache {
public:
    FuncCache(std::function<Result(Args...)> func) : func(func) {}

    Result operator()(Args... args) {
        auto args_tuple = std::make_tuple(args...);
        auto it = cache.find(args_tuple);
        if (it != cache.end()) {
            return it->second;
        }
        Result result = func(args...);
        cache[args_tuple] = result;
        return result;
    }

private:
    std::function<Result(Args...)> func;

    struct TupleHash {
        template <typename... T>
        std::size_t operator()(const std::tuple<T...>& t) const {
            return std::apply([](const T&... args) {
                std::size_t hash = 0;
                ((hash ^= CustomHash<T>()(args) + 0x9e3779b9 + (hash << 6) + (hash >> 2)), ...);
                return hash;
            }, t);
        }

        template <typename T>
        struct CustomHash {
            std::size_t operator()(const T& t) const {
                return std::hash<T>()(t); // Default to std::hash, can be specialized
            }
        };
    };

    std::unordered_map<std::tuple<Args...>, Result, TupleHash> cache;
};