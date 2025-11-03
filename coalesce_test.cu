/* Testing coalescing
    
   Global memory is the main memory of the device, and can be very slow. 
   To make it more efficient, you can make sure that threads in a group of 32 (warp) are accessing consecutive memory addresses.
   This basically seems like a similar concept to optimising for cache lines on a CPU.
   The GPU will coalesce all of those accesses into a single transaction, hence reading 32 items simultaneously.

   If threads end up accessing scattered addresses, the GPU will have to issue multiple transactions per warp.
 */

#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
//using chrono to test out the CPU times

__global__ void coalescedRead(const float* input, float* output, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        output[i] = input[i];
    }
}

__global__ void uncoalescedRead(const float* input, float* output, int N, int stride)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = i * stride; 
    if (idx < N)
    {
        output[i] = input[idx];
    }
}


int main(int argc, char** argv)
{
    int N = 1 << 24;
    int stride = 32;
    size_t size = N * sizeof(float);

    float* h_in = new float[N];
    float* h_out = new float[N];
    for (int i = 0; i < N; i++)
    {
        h_in[i] = static_cast<float>(i);
    }

    float* d_in;
    float* d_out;

    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads -1) / threads;

    auto start = std::chrono::high_resolution_clock::now();
    coalescedRead<<<blocks, threads>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    auto coalesced_time = std::chrono::duration<double, std::milli>(end - start).count();

    
    start = std::chrono::high_resolution_clock::now();
    uncoalescedRead<<<blocks, threads>>>(d_in, d_out, N, stride);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();

    auto uncoalesced_time = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "Coalesced: " << coalesced_time << ", Uncoalesced:" << uncoalesced_time << "\n";

    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_in;
    delete[] h_out;
}
