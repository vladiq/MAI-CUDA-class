#include <iostream>
#include <vector>
#include <iomanip>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>

#define CSC(call)                                                   \
do {                                                                \
    cudaError_t X = call;                                           \
    if (X != cudaSuccess) {                                         \
        fprintf(stderr, "ERROR: in %s:%d. Message: %s\n",           \
                __FILE__, __LINE__, cudaGetErrorString(X));         \
        exit(0);                                                    \
    }                                                               \
} while(0)

struct comparator {
    __host__ __device__ bool operator()(double a, double b) {
        return std::abs(a) < std::abs(b);
    }
};

__global__ void row_exchange_kernel(double* A, const size_t n, const int i, const int max_idx) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offset_x = gridDim.x * blockDim.x;

    for (int j = idx + i; j < n + 1; j += offset_x) {
        double temp = A[j * n + i];
        A[j * n + i] = A[j * n + max_idx];
        A[j * n + max_idx] = temp;
    }
}

__global__ void forward_elimination_kernel(double* A, const size_t n, const int i) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offset_x = gridDim.x * blockDim.x;
    int offset_y = gridDim.y * blockDim.y;

    for (int j = idy + i + 1; j < n + 1; j += offset_y) {
        for (int k = idx + i + 1; k < n; k += offset_x) {
            A[j * n + k] -= A[j * n + i] * A[i * n + k];
        }
    }
}

__global__ void row_normalization_kernel(double* A, const size_t n, const int i) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offset_x = gridDim.x * blockDim.x;

    double p = A[i * n + i];

    for (int j = idx + i + 1; j < n + 1; j += offset_x) {
        A[j * n + i] /= p;
    }
}

__global__ void backward_kernel(const double* A, double* x, const size_t n, const int i) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = gridDim.x * blockDim.x;

    for (int j = i - idx - 1; j >= 0; j -= offset) {
        x[j] -= A[i * n + j] * x[i];
    }
}

__host__ void solve_gaussian(double* A, const size_t n) {
    double* A_dev;
    CSC(cudaMalloc((void **)&A_dev, sizeof(double) * n * (n + 1)));
    CSC(cudaMemcpy(A_dev, A, sizeof(double) * n * (n + 1), cudaMemcpyHostToDevice));

    comparator comp;
    for (int i = 0; i < n; ++i) {
        thrust::device_ptr<double> cur_pointer = thrust::device_pointer_cast(A_dev + i * n);
        thrust::device_ptr<double> max_pointer = thrust::max_element(cur_pointer + i, cur_pointer + n, comp);
        int max_idx = max_pointer - cur_pointer;

        if (max_idx > i) {
            row_exchange_kernel<<<256, 256>>>(A_dev, n, i, max_idx);
            CSC(cudaGetLastError());
        }

        row_normalization_kernel<<<256, 256>>>(A_dev, n, i);
        CSC(cudaGetLastError());

        dim3 grid(32, 32);
        dim3 block(32, 32);
        forward_elimination_kernel<<<grid, block>>>(A_dev, n, i);
        CSC(cudaGetLastError());    
    }

    CSC(cudaMemcpy(A, A_dev, sizeof(double) * n * (n + 1), cudaMemcpyDeviceToHost));

    double* x = new double[n];
    for (int i = 0; i < n; ++i){
        x[i] = A[n * n + i];
    }

    double* x_dev;
    CSC(cudaMalloc((void**)&x_dev, sizeof(double) * n));
    CSC(cudaMemcpy(x_dev, x, sizeof(double) * n, cudaMemcpyHostToDevice));

    for (int i = n - 1; i > 0; --i) {
        backward_kernel<<<256, 256>>>(A_dev, x_dev, n, i);
        CSC(cudaGetLastError());
    }

    CSC(cudaMemcpy(x, x_dev, sizeof(double) * n, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < n; ++i) {
        std::cout << std::setprecision(10) << std::scientific << x[i] << ' ';
    }
    putchar('\n');

    CSC(cudaFree(A_dev));
    CSC(cudaFree(x_dev));
    delete[] x;
}

int main() {
    int n;
    std::cin >> n;
    double* A = new double[n * n + n];

    // merge A and b for convenience
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cin >> A[j * n + i];
        }
    }
    for (int i = 0; i < n; ++i) {
        std::cin >> A[n * n + i];
    }

    solve_gaussian(A, n);
    
    delete[] A;
}