/* Matrix Multiplication
    
   This is the foundation of basically all linear algebra on the GPU.

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

__global__ void matrixMult(const float* A, const float* B, float* C, int M, int N, int K)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N)
  {
      float sum = 0.0f;
      for (int k = 0; k < K; k++)
      {
          sum += A[row * N + col] + B[row * N + col];
      }
      C[row * N + col] = sum;
  }

}

int main()
{
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  //3D matrices
  int M = 512;
  int N = 512;
  int K = 512;
  size_t sizeA = M*K * sizeof(float);
  size_t sizeB = K*N * sizeof(float);
  size_t sizeC = M*N * sizeof(float);

  float *h_A = new float[M*K];
  float *h_B = new float[K*N];
  float *h_C = new float[M*N];

  for (int i = 0; i < M*K; i++) {
      h_A[i] = 1.0f;
  }

  for (int i = 0; i < K*N; i++) {
      h_B[i] = 1.0f;
  }
  float *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc(&d_A, sizeA));
  CUDA_CHECK(cudaMalloc(&d_B, sizeB));
  CUDA_CHECK(cudaMalloc(&d_C, sizeC));

  CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

  dim3 threadsPerBlock(16,16);
  dim3 blocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
               (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

  cudaEventRecord(start);
  matrixMult<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
  cudaEventRecord(stop);

  CUDA_CHECK(cudaEventSynchronize(stop));

  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));

  std::cout << "value of first element in C: " << h_C[0] << "\n";
  std::cout << "value of last element in C: " << h_C[M*N-1] << "\n";

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
