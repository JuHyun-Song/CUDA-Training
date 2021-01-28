#include <cstdio>
#include <iostream>

//Kernel program for the device(GPU): compiled by NVCC

__global__ void mulKernel(int*c, const int* a, const int*b,const int WIDTH){
	int x = threadIdx.x;
	int y = threadIdx.y;
	int i = y * WIDTH + x; //[y][x] = y * WIDTH + x;
	int sum = 0;
	for(int k = 0; k < WIDTH; ++k){
		sum += a[y * WIDTH + k] * b[k * WIDTH + x];
	}
	c[i] = sum;
}


void cpuCode(){

    //host - side data
	const int WIDTH = 5;
	int a[WIDTH][WIDTH];
	int b[WIDTH][WIDTH];
	int c[WIDTH][WIDTH] = {0};
	
	//make matrices A,B
	for(int y = 0; y < WIDTH; ++y){
		for(int x = 0; x < WIDTH; ++x){
			a[y][x] = y + x;
			b[y][x] = y + x;
		}
	}
	
	//calculation code
	for(int y=0; y < WIDTH; ++y){
		for(int x = 0; x < WIDTH; ++x){
			int sum = 0;
			for (int k = 0; k < WIDTH ; ++k){
				sum += a[y][k]*b[k][x];
			}
			c[y][x] = sum;
		}
	}
	
	//print the result
	for(int y = 0; y < WIDTH ; ++y){
		for(int x = 0 ; x < WIDTH ; ++x){
			printf("%5d",c[y][x]);
		}
		printf("\n");
	}

}

void cudaCode(){

    //host-side data
	const int WIDTH = 5;
	int a[WIDTH][WIDTH];
	int b[WIDTH][WIDTH];
	int c[WIDTH][WIDTH] = { 0 };
	
	//make a ,b matrices
	for(int y = 0; y < WIDTH; ++y){
		for(int x = 0; x < WIDTH; ++x){
			a[y][x] = y + x;
			b[y][x] = y + x;
		}
	}
	
	//allocate memory on the device
	//device-side data
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	
	//allocate device memory
	cudaMalloc((void**)&dev_a, WIDTH * WIDTH * sizeof(int));
	cudaMalloc((void**)&dev_b, WIDTH * WIDTH * sizeof(int));
	cudaMalloc((void**)&dev_c, WIDTH*WIDTH * sizeof(int));
	
	//copy form host to device
	cudaMemcpy(dev_a,a,WIDTH * WIDTH * sizeof(int),cudaMemcpyHostToDevice); //dev_a = a;
	cudaMemcpy(dev_b, b,WIDTH * WIDTH * sizeof(int),cudaMemcpyHostToDevice); // dev_b = b;
	
	//launcn a kernel on the GPU with one thread for each element
	dim3 dimBlock(WIDTH , WIDTH, 1 ); // x, y, z
	mulKernel<<<1 , dimBlock>>> (dev_c, dev_a, dev_b, WIDTH);
	//CUDA_CHECK(cudaPeekAtLastError());

	//copy from device to host
	cudaMemcpy(c,dev_c,WIDTH * WIDTH * sizeof(int),cudaMemcpyDeviceToHost); //c = dev_c;

	//free device memory
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);
	
	//print the result
	for(int y = 0; y < WIDTH ; ++y){
		for(int x = 0 ; x < WIDTH ; ++x){
			printf("%d ",c[y][x]);
		}
		printf("\n");
	}

}




int main(void){

    cpuCode();
    std::cout << "+++++"<<std::endl;
    cudaCode();
	
	
	
	return 0;
}



