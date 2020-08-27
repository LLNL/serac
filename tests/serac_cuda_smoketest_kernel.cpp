

__global__ void
    vector_add_kernel(float* out, float* a, float* b, int n)
{
  for (int i = 0; i < n; i++) {
    out[i] = a[i] + b[i];
  }
}


void vector_add(float* out, float* a, float* b, int n)
{
  vector_add_kernel<<<1, 1>>>(out, a, b, n);
}
