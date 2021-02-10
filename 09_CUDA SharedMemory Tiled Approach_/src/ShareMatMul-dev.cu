#include <cstdio>
#include <stdlib.h>//for rand(),malloc(),free()
#include <windows.h>//for QueryPerformanceCounter()

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

const int WIDTH = 1024; //total width is 1024*1024
const int TILE_WIDTH = 32; //block will be (TILE_WIDTH,TILE_WIDTH)
const int GRID_WIDTH = (WIDTH / TILE_WIDTH);//grid will be (GRID_WIDTH,GRID_WIDTH)


__global__ void matmul(float* g_C, const float* g_A, const float* g_B, const int width) {
	__shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
	__shared__ float s_B[TILE_WIDTH][TILE_WIDTH];
	int by = blockIdx.y;
	int bx = blockIdx.x;
	int ty = threadIdx.y;
	int tx = threadIdx.x;
	int gy = by * TILE_WIDTH + ty; //global y index
	int gx = bx * TILE_WIDTH + tx; //global x index
	float sum = 0.0F;
	for (register int m = 0; m < width / TILE_WIDTH; ++m) {
		//read into the shared memory blocks
		s_A[ty][tx] = g_A[gy * width + (m * TILE_WIDTH + tx)];
		s_B[ty][tx] = g_B[(m * TILE_WIDTH + ty) * width + gx];
		__syncthreads();
		//use the shared memory blocks to get the partial sum
		for (register int k = 0; k < TILE_WIDTH; ++k) {
			sum += s_A[ty][k] * s_B[k][tx];
		}
		__syncthreads();
	}
	g_C[gy * width + gx] = sum;
}

//random data generation
void getData(float* ptr, unsigned int size) {
	while (size--) {
		*ptr++ = (float)(rand() % 1000) / 1000.0F;
	}
}

int main(void) {
	float* pA = NULL;
	float* pB = NULL;
	float* pC = NULL;
	long long cntStart, cntEnd, freq;
	QueryPerformanceFrequency((LARGE_INTEGER*)(&freq));

	//malloc memories on the host-side
	pA = (float*)malloc(WIDTH * WIDTH * sizeof(float));
	pB = (float*)malloc(WIDTH * WIDTH * sizeof(float));
	pC = (float*)malloc(WIDTH * WIDTH * sizeof(float));

	//generate sorce data
	getData(pA, WIDTH * WIDTH);
	getData(pB, WIDTH * WIDTH);

	//CUDA:allocate device memory
	float* pAdev = NULL;
	float* pBdev = NULL;
	float* pCdev = NULL;
	cudaMalloc((void**)&pAdev, WIDTH * WIDTH * sizeof(float));
	cudaMalloc((void**)&pBdev, WIDTH * WIDTH * sizeof(float));
	cudaMalloc((void**)&pCdev, WIDTH * WIDTH * sizeof(float));

	//copy from host to device
	cudaMemcpy(pAdev, pA, WIDTH * WIDTH * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(pBdev, pB, WIDTH * WIDTH * sizeof(float), cudaMemcpyHostToDevice);

	//start the timer
	QueryPerformanceCounter((LARGE_INTEGER*)(&cntStart)); //start the stop watch

	//CUDA: launch the kernel
	dim3 dimGrid(GRID_WIDTH, GRID_WIDTH, 1);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	matmul << <dimGrid, dimBlock >> > (pCdev, pAdev, pBdev, WIDTH);

	//end the timer
	QueryPerformanceCounter((LARGE_INTEGER*)(&cntEnd)); //end the stop watch 
	printf("elapsed time = %f msec \n", (double)(cntEnd - cntStart) * 1000.0 / (double)(freq));

	//copy form device to host
	cudaMemcpy(pC, pCdev, WIDTH * WIDTH * sizeof(float), cudaMemcpyDeviceToHost);

	//free device memory
	cudaFree(pAdev);
	cudaFree(pBdev);
	cudaFree(pCdev);

	//print sample cases
	int i = 0;
	int j = 0;
	printf("c[%4d][%4d] = %f\n", i, j, pC[i * WIDTH + j]);

	i = WIDTH / 2;
	j = WIDTH / 2;
	printf("c[%4d][%4d] = %f\n", i, j, pC[i * WIDTH + j]);

	i = WIDTH - 1;
	j = WIDTH - 1;
	printf("c[%4d][%4d] = %f\n", i, j, pC[i * WIDTH + j]);



	return 0;

}
