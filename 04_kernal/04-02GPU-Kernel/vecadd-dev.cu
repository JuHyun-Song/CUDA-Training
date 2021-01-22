#include <cstdio>
#include <stdio.h>

//kernel program for the device (GPU): compiled by NVcc
__global__ void addKernel(int*c, const int *a, const int *b){
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}
#define SIZE 1000000

int main(void){//create 

    int *a = new int[SIZE]; 
    int *b = new int[SIZE];
    int *c = new int[SIZE];
    
    for(int postion = 0 ; postion < SIZE; postion++){
        a[postion] = 2;
        b[postion] = 2;
    }
    
    //device-side data
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    
    //allocate device memory
    cudaMalloc((void**)&dev_a, SIZE * sizeof(int));
    cudaMalloc((void**)&dev_b, SIZE * sizeof(int));
    cudaMalloc((void**)&dev_c, SIZE * sizeof(int));

    //copy from host to device
    cudaMemcpy(dev_a, a, SIZE * sizeof(int), cudaMemcpyHostToDevice);// dev_a = a;
    cudaMemcpy(dev_b, b, SIZE * sizeof(int), cudaMemcpyHostToDevice);// dev_b = b;
    cudaMemcpy(dev_c, b, SIZE * sizeof(int), cudaMemcpyHostToDevice);// dev_b = b;
    
    //launch a Kernel on the GPU with one thread for each element.
    addKernel<<<1,SIZE>>>(dev_c, dev_a, dev_b);

    //copy from device to host
    cudaMemcpy(c, dev_c, SIZE * sizeof(int), cudaMemcpyDeviceToHost);// c = dev_c;

    //free device memory
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    //print the result
    printf("{%d,%d,%d,%d,%d} + {%d,%d,%d,%d,%d}"
    "={%d,%d,%d,%d,%d}\n",
    a[0],a[1],a[2],a[3],a[4],
    b[0],b[1],b[2],b[3],b[4],
    c[0],c[1],c[2],c[3],c[4]);

    return 0;
}

