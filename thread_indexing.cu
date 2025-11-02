/*
    Thread indexing for arrays and matrices
    
    Calculate global thread ID like seen before, using the matmul approach.
    You'd think that you could just do N/threadsPerBlock to get the block num, but it is best practice 
    to have an extra block just in case, hence the ceiling calculation.
    
    With matrices you can use x and y. Hence, a 16x16 matrix representing a block could have 256 threads.
    row is given by blockIdx.y * blockDim.y + threadIdx.y
    column is given by blockIdx.x * blockDim.x + threadIdx.x
    
    Now to extend the vector addition code to matrix addition.

 */

#include <iostream>
#include <cuda_runtime.h>

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort=true)
{
  if (code != cudaSuccess)
  {
    std::cerr << "gpuAssert: " << cudaGetErrorString(code) << ", file: " << file << ", line: " << line << "\n";
    if (abort)
    {
      exit(code);
    }
  }
}

__global__ void matrixAdd(const float* A, const float* B, float* C, int M, int N)
{
  //compute thread identifier, which is also the index of the item it will act on
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N)
  {
    C[row * N + col] = A[row * N + col] + B[row * N + col];
  }

}

int main()
{
  //cuda timing
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  //represent matrices as 1D flattened arrays
  int M = 8;
  int N = 8;
  int length =  M * N;
  size_t size = length * sizeof(float);

  //now we need to do the hard part, which is allocating the memory on the gpu and telling it to run the specific kernel, then copying the results back
  float *h_A = new float[length];
  float *h_B = new float[length];
  float *h_C = new float[length];

  for (int i = 0; i < length; i++) {
      h_A[i] = 1.0f;
      h_B[i] = 2.0f;
  }

  float *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc(&d_A, size));
  CUDA_CHECK(cudaMalloc(&d_B, size));
  CUDA_CHECK(cudaMalloc(&d_C, size));

  CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

  int threadsPerBlock = 256;
  int blocks = (length + threadsPerBlock - 1) / threadsPerBlock;

  cudaEventRecord(start);
  matrixAdd<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N);
  cudaEventRecord(stop);

  CUDA_CHECK(cudaEventSynchronize(stop));

  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

  std::cout << "value of first element in C: " << h_C[0] << "\n";

  //timing results
  float milliseconds = 0;

  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Timing (ms): " << milliseconds << "\n"; 

  delete[] h_A;
  delete[] h_B;
  delete[] h_C;
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  return 0;
}
