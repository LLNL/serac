_

#include <gtest/gtest.h>

#include <algorithm>

    _global__ void
    vector_add_kernel(float* out, float* a, float* b, int n)
{
  for (int i = 0; i < n; i++) {
    out[i] = a[i] + b[i];
  }
}

int main()
{
  constexpr int N = 1000;

  float a[N];
  float b[N];
  float out[N];

  std::fill(a, a + N, 2.0f);
  std::fill(b, b + N, 4.0f);

  vector_add_kernel<<<1, 1>>>(out, a, b, N);

  EXPECT_TRUE(std::all_of(c, c + N, [](const float f) { return f == 6.0; }));
}
