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
 //copied the below version from before 
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

//using the naive matmul approach for now
__global__ void matmul(const float* A, const float* B, float* C, int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N)
    {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k)
        {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
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
    //output
    float h_Y[M*N];

    float *d_X;
    float *d_W;
    float *d_b;
    float *d_Y;

    cudaMalloc(&d_X, M*K * sizeof(float));
    cudaMalloc(&d_W, K*N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_Y, M*N * sizeof(float));

    cudaMemcpy(d_X, h_X, M*K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, h_W, K*N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(16,16);
    dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);

    //forward pass here
    matmul<<<blocks, threads>>>(d_X, d_W,d_Y, M, N, K);
    addBias<<<blocks, threads>>>(d_Y, d_b, M, N);
    reluActivation<<<blocks, threads>>>(d_Y, M*N);

    cudaMemcpy(h_Y, d_Y, M*N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_X);
    cudaFree(d_W);
    cudaFree(d_b);
    cudaFree(d_Y);

    std::cout << "output after relu: \n";
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            std::cout << h_Y[i*N + j] << " ";
        }
        std::cout << "\n";
    }
    

}
