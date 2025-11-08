/* MLP Forward Pass
    
   MLPs are just matrix computations, like Y = f(XW + b):
    - Y is the output (batch size * output dimension)
    - f is the activation function, like sigmoid or ReLU
    - X is the input matrix
    - W is the weights matrix
    - b is the bias matrix
    
   There are 3 kernels needed here:
    - applying activation function
    - matmul for XW
    - addition for bias

   This is so much more efficient than my stupid java coursework implementation of an MLP
   although this doesn't do any backprop yet.
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

}

__global__ void addBias(float* C, const float* bias, int M, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    //add bias to each row of the matrix
    if (row < M && col < N)
    {
        C[row * N + col] += bias[col];
    }
}

__global__ void reluActivation(float* C, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ((i < size) && (C[i] < 0.0f)) 
    {
        C[i] = 0.0f;
    }
}

int main(int argc, char** argv)
{
    //batch size, input features, output features
    const int M = 4;
    const int K = 3;
    const int N = 2;
    
    float h_X[M*K] = {1,2,3,
                      4,5,6,
                      7,8,9,
                      10,11,12};

    float h_W[K*N] = {0.2f, 0.8f,
                      0.5f, 0.3f,
                      0.9f, 0.4f};

    float h_b[N] = {1.0f, -1.0f};


}
