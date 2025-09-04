#include <hostutils/utils.hpp>
#include <kernels/add_vec.cuh>
#include <print>

#define SIZE 4096

int main(int argc, char *argv[]) {
  say_hello();
  float *a = new float[SIZE];
  float *b = new float[SIZE];
  float *c = new float[SIZE];

  for (int i = 0; i < SIZE; i++) {
    a[i] = static_cast<float>(i);
    b[i] = SIZE - static_cast<float>(i);
  }
  gpu::add_vec(c, a, b, SIZE);

  for (int i = 0; i < SIZE; i++) {
    std::print("{} ", c[i]);
  }

  delete[] a;
  delete[] b;
  delete[] c;
}
