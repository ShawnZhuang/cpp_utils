
#include <gtest/gtest.h>
#include <shift_stack.h>

// 测试 shift_stack 函数的正确性
TEST(ShiftStackTest, BasicAssertions)
{
    ShiftStack<int, std::equal_to<int>> ss;
    ss.push(1);
    ss.push(2);
    ss.push(3);
    ss.push(4);
    auto status = ss.setTop(3);
    EXPECT_TRUE(status);
    EXPECT_EQ(ss.top(), 3);
    status ss.setTop(6);
    EXPECT_FALSE(status);
    EXPECT_EQ(ss.top(), 3);
}
