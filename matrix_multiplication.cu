/* Matrix Multiplication
    
   This is the foundation of basically all linear algebra on the GPU.
   In the below implementation we initially read the full row and column from the matrix within each thread, which can be slow.
   That's why we can used advanced techniques such as using shared memory and tiling.

   Shared memory is like on chip cache that threads in a block can share.
   Matrices need to be split into tiles that fit into shared memory. Each thread in a block computes 1 element of the tile.
   Threads can work together to load tiles into shaed memory.
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

#define TILE_WIDTH 16
  
__global__ void matrixMult(const float* A, const float* B, float* C, int M, int N, int K)
{      
  __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
  __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];
         
  int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
  int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
  

  float value = 0.0f;
  for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; t++)  
  {

      if (row < M && (t * TILE_WIDTH + threadIdx.x) < K)
      {  
          tileA[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_WIDTH + threadIdx.x];  
      } 
      else  
      {
          tileA[threadIdx.y][threadIdx.x] = 0.0f; 
      } 
      
      if ((t * TILE_WIDTH + threadIdx.y) < K && col < N)
      {
          tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_WIDTH + threadIdx.y) * N + col];
      }
      else
      {
          tileB[threadIdx.y][threadIdx.x] = 0.0f;
      }

      //wait for all of the threads to load the data
      __syncthreads();

      for (int k = 0; k < TILE_WIDTH; k++)
      {
          value += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
      }

      __syncthreads();
    
  }       

  if (row < M && col < N)
  {
      C[row * N + col] = value;
  }

/*   if (row < M && col < N) */
/*   { */  
/*       float sum = 0.0f; */
/*       for (int k = 0; k < K; k++) */
/*       { */  
/*           sum += A[row * N + col] + B[row * N + col]; */
/*       } */
/*       C[row * N + col] = sum; */
/*   } */

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

  dim3 threadsPerBlock(TILE_WIDTH,TILE_WIDTH);
  dim3 blocks((N + TILE_WIDTH - 1) / TILE_WIDTH,
               (M + TILE_WIDTH - 1) / TILE_WIDTH);

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
