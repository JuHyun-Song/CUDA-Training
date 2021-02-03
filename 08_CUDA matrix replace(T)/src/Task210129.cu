#include<cstdio>
#include <iostream>

const int WIDTH = 2000;
const int HEIGHT = 1000;
const int SIZE = WIDTH * HEIGHT;

// kernel program for the dervice (GPU):compiled by NVCC
__global__ void oppositeKernel(int *dev_numbers, int *dev_numbersCopy) {

	/*for(int y = 0; y < HEIGHT; y++){
        for(int x = 0; x < WIDTH; x++){
            dev_numbersCopy[y*WIDTH+x]=dev_numbers[SIZE-(y*WIDTH+x)-1];
        }
    }*/
    int index = threadIdx.x;
    int stride = blockDim.x;
    for (int i = index; i < SIZE; i += stride){
        int row = i / WIDTH;
        int col = i % WIDTH;
        dev_numbersCopy[row * WIDTH + col] = dev_numbers[row * WIDTH + WIDTH - col - 1];
    }
}

//opposite the postion of Array elements in host side
void oppositeArr(int **array1 ,int **array2){

	for(int yCounter = 0; yCounter < HEIGHT; yCounter++){
        for(int xCounter = 0; xCounter < WIDTH; xCounter++){
            // copy
            array2[yCounter][xCounter] = array1[yCounter][xCounter];
        }
           
	    for(int xCounter = 0; xCounter < WIDTH; xCounter++){   
            array1[yCounter][xCounter]=array2[yCounter][WIDTH-xCounter-1];
	    } 
    }
}  

//print the elements of array
void showArr(int *array[WIDTH] , int row, int col){
	for(int y = 0; y < row ; ++y){
		for(int x = 0 ; x < col ; ++x){
			printf("%d ",array[y][x]);
		}
		printf("\n");
	}
}

void showArr1d(int *array , int row, int col){
	for(int y = 0; y < row ; ++y){
		for(int x = 0 ; x < col ; ++x){
			printf("%d ",array[y * WIDTH + x]);
		}
		printf("\n");
	}
}     

int main(void) {

    //make opposite Arr in host side
    //oppositeArr(numbers,bArr);

	// host-side array data
	int* numbers = new int[SIZE];// next step: int[HEIGHT][WIDTH]
	
    // fill out the content of each line by start 4 , 5 ...
	for(int yCounter = 0; yCounter < HEIGHT; yCounter++){
	    for(int xCounter = 0; xCounter < WIDTH; xCounter++){
	        numbers[yCounter * WIDTH + xCounter]=xCounter + 4; //numbers[][]=4,5,6,7,8.....1999  
	    }
    }
    
    //show the elements of array
	printf("before using cuda\n");
	showArr1d(numbers,10,30);
	printf("\n");
	
	// CUDA below //
	
	// device-side array data
	int* dev_numbers = 0;
	int* dev_numbersCopy = 0;

	// allocate device memory
	cudaMalloc((void**)&dev_numbers,     WIDTH * HEIGHT * sizeof(int));
	cudaMalloc((void**)&dev_numbersCopy, WIDTH * HEIGHT * sizeof(int));
	
	
	// copy from host to device
	cudaMemcpy(dev_numbers, numbers, WIDTH * HEIGHT* sizeof(int),cudaMemcpyHostToDevice); // dev_numbers = numbers;
	

	// launch a kernel on the GPU with one thread for each element
	//dim3 dimGrid(1, 1, 1);
	//dim3 dimBlock(WIDTH, HEIGHT, 1); // x,y,z
	oppositeKernel <<< 1, 512 >>> (dev_numbers, dev_numbersCopy);
	
	// Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
    
    // copy from device to host 
	cudaMemcpy(numbers, dev_numbersCopy, WIDTH * HEIGHT * sizeof(int), cudaMemcpyDeviceToHost); 

    //show the elements of array
	printf("after using cuda\n");
	showArr1d(numbers,3,50);
	printf("\n");

    delete []numbers;
    numbers = nullptr;
    
    // free device memory
	cudaFree(dev_numbers);
	cudaFree(dev_numbersCopy);

    
    
    return 0;

}
