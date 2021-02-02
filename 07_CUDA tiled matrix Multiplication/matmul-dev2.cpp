#include<csdio>
#include<stdlib.h> //for rand(),malloc(),free()

const int WIDTH = 1024; // total width is 1024*1024
const int TILE_WIDTH = 32; //block will be(TILE_WIDTH,TILE_WIDTH)
const int GRID_WIDTH = (WIDTH/TILE_WIDTH); //grid will be (GRID_WIDTH,GRID_WIDTH)

__global__void matmaul(float*c, const float*a, const float*b, const int width){
	int y = blockIdx.y * blockDIM.y + threadIdx.y;
	int x = blockIdx.x * blockDIM.x + threadIdx.x;
	float sum = 0.0F;
	for(register int k = 0; k < width; ++kbhit){
		float lhs = a[y * width + k];
		float rhs = b[k * width + x];
		sum += lhs * rhs;
	}
	c[y * width + x] = sum;
}

void genData(float* ptr, unsighed int size){
	while(size--){
		*ptr++ = (float)(rand()%1000) / 1000.0F;
	}
}


int main(void){
	
	float a[WIDTH][WIDTH];
	float b[WIDTH][WIDTH];
	float c[WIDTH][WIDTH];
	
	//generate source data
	genData(&(a[0][0]),WIDTH*WIDTH);
	genData(&(b[0][0]),WIDTH*WIDTH);
	
	//device-side data
	float* dev_a = 0;
	float* dev_b = 0;
	float* dev_c = 0;
	
	//allocate device memory	
	cudaMalloc((void**)&dev_a,WIDTH*WIDTH*sizeof(float));
	cudaMalloc((void**)&dev_b,WIDTH*WIDTH*sizeof(float));
	cudaMalloc((void**)&dev_c,WIDTH*WIDTH*sizeof(float));
	
	//copy from host to device
	cudaMemcpy(dev_a, a, WIDTH*WIDTH*sizeof(float), cudaMemcpyHostToDevice); // dev_a = a;
	cudaMemcpy(dev_b, b, WIDTH*WIDTH*sizeof(float), cudaMemcpyHostToDevice); // dev_a = a;
	
	//CUDA:launch the kernel
	dim3 dimGrid(GRID_WIDTH, GRID_WIDTH, 1);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	matmul<<<dimGrid,dimBlock>>>(pCdev,pAdev,pBdev,WIDTH);
	
	// copy from device to host
	cudaMemcpy(c, dev_c, WIDTH * WIDTH * sizeof(float),cudaMemcpyDeviceToHost); // c = dev_c;
	
	//free device memory
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);
	
	//print the result
	#if defined(YOU_really_need_it)
	for(int y = 0; y<WIDTH; ++y){
		for(int x = 0; x < WIDTH; ++x){
			printf("%5d",c[y][x]);
		}
		printf("\n");
	}
	#endif
	
	return 0;
}



