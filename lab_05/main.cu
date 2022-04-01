#include <iostream>
#include <cstdio>
#include <cstring>
#include <vector>
#include <limits>
#include <algorithm>

#define CSC(call)                                                   \
do {                                                                \
    cudaError_t X = call;                                           \
    if (X != cudaSuccess) {                                         \
        fprintf(stderr, "ERROR: in %s:%d. Message: %s\n",           \
                __FILE__, __LINE__, cudaGetErrorString(X));         \
        exit(0);                                                    \
    }                                                               \
} while(0)


#define ELEMENTS_PER_BLOCK 512u
#define NUM_BANKS 32u
#define LOG_NUM_BANKS 5u
#define MAX_ELEM_VALUE 17777216u
#define CONFLICT_FREE_OFFSET(n) ((n) + ((n) >> LOG_NUM_BANKS))

__global__ void histogram(unsigned* C_dev, const int* data_dev, const unsigned n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;

    while (idx < n) {
        atomicAdd(C_dev + data_dev[idx], 1);
        idx += offset;
    }
}

__global__ void scan_device(unsigned* data, unsigned* sums, const unsigned n) {

	extern __shared__ int temp[];

	int tid = threadIdx.x;
	int blockID = blockIdx.x;
	int blockOffset = ELEMENTS_PER_BLOCK * blockID;

	int ai = tid;
	int bi = tid + ELEMENTS_PER_BLOCK / 2;
	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

	if (tid < n) {
		temp[bankOffsetA] = data[blockOffset + ai];
		temp[bankOffsetB] =	data[blockOffset + bi];
	} else {
		temp[bankOffsetA] = 0;	
		temp[bankOffsetB] = 0;
	}

	unsigned offset = 1;

	for (int d = ELEMENTS_PER_BLOCK / 2; d > 0; d >>= 1) {
		int l = offset * ((tid << 1) + 1) - 1;
		int r = offset * ((tid << 1) + 2) - 1;

		l = CONFLICT_FREE_OFFSET(l);
		r = CONFLICT_FREE_OFFSET(r);
		
		__syncthreads();
		if (tid < d) {
			temp[r] += temp[l];
		}

		offset *= 2;
	}

	if (tid == 0) {
		int last_elem = CONFLICT_FREE_OFFSET(ELEMENTS_PER_BLOCK - 1);
		sums[blockID] = temp[last_elem];
		temp[last_elem] = 0;
	}

	for (int d = 1; d <= ELEMENTS_PER_BLOCK / 2; d <<= 1) {
		offset >>= 1;

		int l = offset * ((tid << 1) + 1) - 1;
		int r = offset * ((tid << 1) + 2) - 1;

		l = CONFLICT_FREE_OFFSET(l);
		r = CONFLICT_FREE_OFFSET(r);

		__syncthreads();
		if (tid < d) {
			unsigned t = temp[l];
			temp[l] = temp[r];
			temp[r] += t;
		}
	}
	__syncthreads();

	if (tid < n) {
		data[blockOffset + ai] = temp[bankOffsetA];
		data[blockOffset + bi] = temp[bankOffsetB];
	}
}

__global__ void add(unsigned* data, unsigned* sums, const unsigned n) {
	int tid = threadIdx.x;
	int blockID = blockIdx.x;
	int blockOffset = ELEMENTS_PER_BLOCK * blockID;
	
	if (blockOffset + tid < n) {
		data[blockOffset + tid] += sums[blockID];
	}
}

__host__ void scan(unsigned* C_dev, const unsigned n) {
	unsigned n_blocks = std::max(1u, n / ELEMENTS_PER_BLOCK);
	unsigned shmem_size = ELEMENTS_PER_BLOCK * sizeof(unsigned);
	
	unsigned* sums;
	CSC(cudaMalloc(&sums, n_blocks * sizeof(unsigned)));

	scan_device<<<n_blocks, ELEMENTS_PER_BLOCK / 2, 2 * shmem_size>>>(
		C_dev, sums, n
	);
	CSC(cudaGetLastError());
	
	if (n_blocks == 1) {
		CSC(cudaFree(sums));
		return;
	}

	scan(sums, n_blocks);

	add<<<n_blocks, ELEMENTS_PER_BLOCK>>>(
		C_dev, sums, n
	);
	CSC(cudaGetLastError());

	CSC(cudaFree(sums));
}

__global__ void last_step(
	int* data, const unsigned* C, const unsigned len_C
){
    int idx = blockDim.x * blockIdx.x +  threadIdx.x;
    int offset = blockDim.x * gridDim.x;

    for (int tid = idx; tid < len_C; tid += offset){
        int low = 0;
        if (tid != 0) {
            low = C[tid - 1];
        }

        for (int i = C[tid] - 1; i >= low; --i){
            data[i] = tid - 1;
        }
    }
}

__host__ void counting_sort(const unsigned n, int* data) {
    unsigned* C_dev;
    int* data_dev;

    CSC(cudaMalloc((void**) &data_dev, n * sizeof(int)));
    CSC(cudaMalloc((void**) &C_dev, MAX_ELEM_VALUE * sizeof(unsigned)));

    CSC(cudaMemcpy(data_dev, data, n * sizeof(int), cudaMemcpyHostToDevice));
    CSC(cudaMemset(C_dev, 0, MAX_ELEM_VALUE * sizeof(unsigned)));

	cudaEvent_t start, stop;
	float gpu_time = 0.0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

    histogram<<<256, 256>>>(C_dev, data_dev, n);
    CSC(cudaGetLastError());
    CSC(cudaDeviceSynchronize());

	scan(C_dev, MAX_ELEM_VALUE);

    last_step<<<256, 256>>>(data_dev, C_dev, MAX_ELEM_VALUE);
	CSC(cudaGetLastError());

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpu_time, start, stop);

	std::cout << "time " << gpu_time;

    CSC(cudaMemcpy(data, data_dev, n * sizeof(int), cudaMemcpyDeviceToHost));
}


int main() {
    unsigned n;
    int* data = nullptr;

    try {
        std::freopen(nullptr, "rb", stdin);
        std::fread(&n, sizeof(n), 1, stdin);
        data = new int[n];
        std::fread(data, sizeof(int), n, stdin);
        std::fclose(stdin);
        
		counting_sort(n, data);

        delete[] data;

    } catch(std::exception const& e) {
        std::cerr << e.what() << '\n';
        delete[] data;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}