/* Lesson 1
 * CUDA lets you utilise the thousands of threads that a gpu has in order to do repetitive code rapidly.
 *
 * Grid - collection of blocks
 * Block - group of threads that cooperate using shared memory
 * Thread - smallest unit of execution
 *
 * Each thread knows certain values:
 *  gridDim.x - no. of blocks per grid
 *  blockDim.x - no. of threads per block
 *  blockIdx.x - blocks index in grid
 *  threadIdx.x - index of thread within block
 *
 * This lets a global thread ID be computed:
 * blockIdx.x * blockDim.x + threadIdx.x 
 * This is used to ensure each thread can work on a different bit of data
 *
 * What is a CUDA kernel? It's just a function that runs on a gpu. marked with __global__ identifier at the start.
 */

#include <iostream>
#include <cuda_runtime.h>

__global__ void vectorAdd(const float* A, const float* B, float* C, int N)
{
  //compute thread identifier, which is also the index of the item it will act on
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < N)
  {
    C[i] = A[i] + B[i];
  }

}

int main(int argc, char** argv)
{
  //Testing out adding vectors of a million floats
  int N = 1 << 20;
  size_t size = N * sizeof(float);
  
  float* h_A = new float[N];
  float* h_B = new float[N];
  float* h_C = new float[N];

  for (int i = 0; i < N; i++)
  {
    h_A[i] = 1.0f;
    h_B[i] = 2.0f;
  }
  
  //now we need to do the hard part, which is allocating the memory on the gpu and telling it to run the specific kernel, then copying the results back

  float* d_A;
  float* d_B;
  float* d_C;
  cudaMalloc(&d_A, size);
  cudaMalloc(&d_B, size);
  cudaMalloc(&d_C, size);

  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_C, size, cudaMemcpyHostToDevice);

  int threadsPerBlock = 256;
  int blocks = (N + threadsPerBlock -1) / threadsPerBlock;

  vectorAdd<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

  cudaDeviceSynchronize();

  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  std::cout << "value of first element in C: " << h_C[0] << "\n";

  delete[] h_A;
  delete[] h_B;
  delete[] h_C;
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  return 0;
}
