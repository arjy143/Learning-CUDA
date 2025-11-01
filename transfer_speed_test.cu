/* Testing transfer speed between CPU and GPU
 * GPU VRAM is video ram, not virtual ram like what it is on CPU. 2 separate things.
 * Lets test the speed of the transfer between cpu and gpu
 */

#include <stdio.h>
#include <cstdlib>
#include <cuda_runtime.h>

int main(int argc, char** argv)
{	
	//by default it will bitshift 24 left to give about 16 million floats (64mb)
	int bitshift = 24;
	
	//can pass in a value to bit shift by
	if (argc > 0)
	{
		bitshift = std::atoi(argv[1]);
	}

	size_t N = 1ULL << bitshift;
	size_t size = N * sizeof(float);

	printf("Amount of floats (N): %zu \n", N);
	printf("size of floats (Bytes): %zu \n", size);

	float *h_data = new float[N];
	for (int i = 0; i < N; i++)
	{
		h_data[i] = i * 0.5f;
	}

	float *d_data;
	cudaMalloc(&d_data, size);

	//timing
	cudaEvent_t start;
	cudaEvent_t stop;
	
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
	cudaEventRecord(stop);
	
	cudaEventSynchronize(stop);
	float time = 0;
	cudaEventElapsedTime(&time, start, stop);

	printf("Elapsed time(ms): %.4f \n", time);

	cudaFree(d_data);
	delete[] h_data;
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return 0;
}

