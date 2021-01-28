#include<cstdio>

//calculate matrix multiplication on CPU
int main(void) {

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
				sum += a[y][k]*b[k]x];
			}
			c[y][x] = sum;
		}
	}
	
	//print the result
	for(int y = 0; y < WIDTH ; ++y){
		for(int x = 0 ; x < WIDTH ; ++x{
			printf(“%5d”,c[y][x]);
		}
		printf(“\n”);
	}


	return 0;
}




