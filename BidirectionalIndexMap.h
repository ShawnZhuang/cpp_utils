#include <iostream>
#include <unordered_map>
#include <stdexcept>

// 假设 ICHECK 宏定义如下（可以根据实际需求调整）
#define ICHECK(condition) \
  if (!(condition)) {     \
    throw std::runtime_error("Check failed: " #condition); \
  }

template <typename Key, typename Value, typename Hash = std::hash<Key>,
          typename KeyEqual = std::equal_to<Key>, typename ValueHash = std::hash<Value>,
          typename ValueEqual = std::equal_to<Value>>
class BidirectionalIndexMap {
public:
  using forward_map_type = std::unordered_map<Key, Value, Hash, KeyEqual>;
  using reverse_map_type = std::unordered_map<Value, Key, ValueHash, ValueEqual>;

private:
  forward_map_type forward_map_;
  reverse_map_type reverse_map_;

public:
  // 添加或更新键值对
  void insert(const Key& key, const Value& value) {
    ICHECK(forward_map_.count(key) == 0 && reverse_map_.count(value) == 0)
        << "Key or value already exists.";
    
    forward_map_[key] = value;
    reverse_map_[value] = key;
  }

  // 删除键值对
  bool erase(const Key& key) {
    auto it = forward_map_.find(key);
    if (it == forward_map_.end()) {
      return false; // 键不存在
    }
    
    Value value = it->second;
    forward_map_.erase(it);
    reverse_map_.erase(value);
    return true;
  }

  // 通过键查找值
  Value get_value(const Key& key) const {
    auto it = forward_map_.find(key);
    ICHECK(it != forward_map_.end()) << "Key not found.";
    return it->second;
  }

  // 通过值查找键
  Key get_key(const Value& value) const {
    auto it = reverse_map_.find(value);
    ICHECK(it != reverse_map_.end()) << "Value not found.";
    return it->second;
  }

  // 检查键是否存在
  bool contains_key(const Key& key) const {
    return forward_map_.count(key) > 0;
  }

  // 检查值是否存在
  bool contains_value(const Value& value) const {
    return reverse_map_.count(value) > 0;
  }

  // 获取正向映射的大小
  std::size_t size() const {
    return forward_map_.size();
  }

  // 清空所有键值对
  void clear() {
    forward_map_.clear();
    reverse_map_.clear();
  }

  // 重载 << 运算符，用于打印整个映射
  friend std::ostream& operator<<(std::ostream& os, const BidirectionalIndexMap& map) {
    os << "Forward Map:\n";
    for (const auto& pair : map.forward_map_) {
      os << pair.first << ": " << pair.second << "\n";
    }
    os << "Reverse Map:\n";
    for (const auto& pair : map.reverse_map_) {
      os << pair.first << ": " << pair.second << "\n";
    }
    return os;
  }
};

// 示例用法
int main() {
  try {
    BidirectionalIndexMap<int, std::string> bidirMap;

    // 插入键值对
    bidirMap.insert(1, "one");
    bidirMap.insert(2, "two");
    bidirMap.insert(3, "three");

    // 打印整个映射
    std::cout << bidirMap << std::endl;

    // 通过键查找值
    std::cout << "Value of key 2: " << bidirMap.get_value(2) << std::endl;

    // 通过值查找键
    std::cout << "Key of value 'three': " << bidirMap.get_key("three") << std::endl;

    // 删除键值对
    bidirMap.erase(1);
    std::cout << "After erasing key 1:\n" << bidirMap << std::endl;

    // 检查键和值是否存在
    std::cout << "Contains key 2: " << bidirMap.contains_key(2) << std::endl;
    std::cout << "Contains value 'four': " << bidirMap.contains_value("four") << std::endl;

    // 清空映射
    bidirMap.clear();
    std::cout << "After clearing:\n" << bidirMap << std::endl;

  } catch (const std::exception& e) {
    std::cerr << "Exception caught: " << e.what() << std::endl;
  }

  return 0;
}
