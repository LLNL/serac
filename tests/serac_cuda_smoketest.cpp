#include <gtest/gtest.h>

#include <algorithm>

void vector_add(float* out, float* a, float* b, int n);

TEST(cuda_smoketest, vec_add)
{
  constexpr int N = 10;

  float a[N];
  float b[N];
  float out[N];

  std::fill(a, a + N, 2.0f);
  std::fill(b, b + N, 4.0f);

  vector_add(out, a, b, N);

  std::for_each(out, out + N, [](const float f) { EXPECT_FLOAT_EQ(f, 6.0); });
}
