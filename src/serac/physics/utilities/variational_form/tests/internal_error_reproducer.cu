#include <tuple>
auto foo(int a, int b) { return std::tuple<int,int>{a, b}; }
int main() { foo(3, 2); }
