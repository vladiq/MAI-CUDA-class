#include <stdio.h>
#include <stdlib.h>
#include <iostream>

__global__ void kernel(double* a, double* b, double* c, int N) {
	int i, idx = blockDim.x * blockIdx.x + threadIdx.x;	
	int offset = blockDim.x * gridDim.x;				
	for(i = idx; i < N; i += offset) {
		c[i] = (a[i] >= b[i]) ? a[i] : b[i];
    }
}

int main() {
	int N;
	std::cin >> N;
	double* a = new double[N];
    double* b = new double[N];
    double* c = new double[N];
    
	for (int i = 0; i < N; ++i) {
		std::cin >> a[i];
	}
	for (int i = 0; i < N; ++i) {
		std::cin >> b[i];
	}

	double *device_a, *device_b, *device_c;

	cudaMalloc((void**) &device_a, N * sizeof(double));
	cudaMalloc((void**) &device_b, N * sizeof(double));
	cudaMalloc((void**) &device_c, N * sizeof(double));

	cudaMemcpy(device_a, a, N * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(device_b, b, N * sizeof(double), cudaMemcpyHostToDevice);

	kernel<<<256, 256>>>(device_a, device_b, device_c, N);

	cudaMemcpy(c, device_c, N * sizeof(double), cudaMemcpyDeviceToHost);

	std::cout.precision(10);
	std::cout.setf(std::ios::scientific);
	for (int i = 0; i < N; i++) {
		std::cout << c[i] << " ";
	}
	putchar('\n');

	cudaFree(device_a);
	cudaFree(device_b);
	cudaFree(device_c);
    delete[] a;
    delete[] b;
    delete[] c;

	return 0;
}