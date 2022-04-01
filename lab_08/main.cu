#include <iostream>
#include <string>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <mpi/mpi.h>

#define CSC(call)  													\
do {																\
	cudaError_t res = call;											\
	if (res != cudaSuccess) {										\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);													\
	}																\
} while(0)

#define _i(i, j, k) (((i) + 1) +                                         \
                    (block_size.x + 2) * ((j) + 1) +                     \
                    (block_size.x + 2) * (block_size.y + 2) * ((k) + 1))

#define _i_ker(i, j, k) (((i) + 1) +                                         \
                        (block_size_x + 2) * ((j) + 1) +                     \
                        (block_size_x + 2) * (block_size_y + 2) * ((k) + 1))

#define _ib(i, j, k) ((i) + n_blocks.x * (j) + n_blocks.x * n_blocks.y * (k))

#define _ibx(id) ((id) % n_blocks.x)
#define _iby(id) (((id) / n_blocks.x) % n_blocks.x)
#define _ibz(id) ((id) / (n_blocks.x * n_blocks.y))

struct NBlocksPerDim {
    NBlocksPerDim() = default;
    NBlocksPerDim(int nbx, int nby, int nbz) : x(nbx), y(nby), z(nbz){}
    int x, y, z;
};

struct BlockSize {
    BlockSize() = default;
    BlockSize(int nx, int ny, int nz) : x(nx), y(ny), z(nz){}
    int x, y, z;
};

struct DimSize {
    DimSize() = default;
    DimSize(double lx, double ly, double lz) : x(lx), y(ly), z(lz){}
    double x, y, z;
};

struct H {
    H() = default;
    H(double hx, double hy, double hz) : x(hx), y(hy), z(hz){}
    double x, y, z;
};

struct BorderCondition {
    BorderCondition() = default;
    BorderCondition(double d, double u, double l, double r, double f, double b) 
        : bottom(d), top(u), left(l), right(r), face(f), back(b){}
    double bottom, top, left, right, face, back;
};

__global__ void kernel_set_border(
    double* data, 
    int block_size_x, 
    int block_size_y, 
    int block_size_z, 
    double bc1, 
    double bc2, 
    char axis
) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    if (axis == 'x') {
        double border_left = bc1, border_right = bc2;

        for (int j = idy; j < block_size_y; j += offsety) {
            for (int k = idx; k < block_size_z; k += offsetx) {
                data[_i_ker(-1, j, k)] = border_left;
                data[_i_ker(block_size_x, j, k)] = border_right;
            }
        }
    } else if (axis == 'y') {
        double border_face = bc1, border_back = bc2;

        for (int i = idy; i < block_size_x; i += offsety) {
            for (int k = idx; k < block_size_z; k += offsetx) {
                data[_i_ker(i, -1, k)] = border_face;
                data[_i_ker(i, block_size_y, k)] = border_back;
            }
        }
    } else if (axis == 'z'){
        double border_bottom = bc1, border_top = bc2;

        for (int i = idy; i < block_size_x; i += offsety) {
            for (int j = idx; j < block_size_y; j += offsetx) {
                data[_i_ker(i, j, -1)] = border_bottom;
                data[_i_ker(i, j, block_size_z)] = border_top;
            }
        }
    }
}

__global__ void kernel_fill_buff_yz(
    double* data, 
    double* buff, 
    int block_size_x, 
    int block_size_y, 
    int block_size_z, 
    int x_value
) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (int j = idy; j < block_size_y; j += offsety) {
        for (int k = idx; k < block_size_z; k += offsetx) {
            buff[k * block_size_y + j] = data[_i_ker(x_value - 1, j, k)];
        }
    }
}

__global__ void kernel_fill_buff_xz(
    double* data,
    double* buff, 
    int block_size_x, 
    int block_size_y, 
    int block_size_z, 
    int y_value
) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (int i = idy; i < block_size_x; i += offsety) {
        for (int k = idx; k < block_size_z; k += offsetx) {
            buff[k * block_size_x + i] = data[_i_ker(i, y_value - 1, k)];
        }
    }
}

__global__ void kernel_fill_buff_xy(
    double* data, 
    double* buff, 
    int block_size_x, 
    int block_size_y, 
    int block_size_z, 
    int z_value
) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (int i = idy; i < block_size_x; i += offsety) {
        for (int j = idx; j < block_size_y; j += offsetx) {
            buff[j * block_size_x + i] = data[_i_ker(i, j, z_value - 1)];
        }
    }
}

__global__ void kernel_fill_data_yz(
    double* data,
    double* buff, 
    int block_size_x, 
    int block_size_y, 
    int block_size_z, 
    int x_value
) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (int j = idy; j < block_size_y; j += offsety) {
        for (int k = idx; k < block_size_z; k += offsetx) {
            data[_i_ker(x_value, j, k)] = buff[k * block_size_y + j];
        }
    }
}

__global__ void kernel_fill_data_xz(
    double* data, 
    double* buff, 
    int block_size_x, 
    int block_size_y, 
    int block_size_z, 
    int y_value
) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (int i = idy; i < block_size_x; i += offsety) {
        for (int k = idx; k < block_size_z; k += offsetx) {
            data[_i_ker(i, y_value, k)] = buff[k * block_size_x + i];
        }
    }
}

__global__ void kernel_fill_data_xy(
    double* data, 
    double* buff, 
    int block_size_x, 
    int block_size_y, 
    int block_size_z, 
    int z_value
) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (int i = idy; i < block_size_x; i += offsety) {
        for (int j = idx; j < block_size_y; j += offsetx) {
            data[_i_ker(i, j, z_value)] = buff[j * block_size_x + i];
        }
    }
}

__global__ void kernel_main_loop(
    double* data, 
    double* next, 
    int block_size_x, 
    int block_size_y, 
    int block_size_z, 
    double hx, 
    double hy,
    double hz
) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int idz = blockDim.z * blockIdx.z + threadIdx.z;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
	int offsetz = blockDim.z * gridDim.z;

    double divisor = 2.0 * (1.0 / (hx * hx) + 1.0 / (hy * hy) + 1.0 / (hz * hz));

    for (int i = idz; i < block_size_x; i += offsetz) {
        for (int j = idy; j < block_size_y; j += offsety) {
            for (int k = idx; k < block_size_z; k += offsetx) {
                next[_i_ker(i, j, k)] = (
                        (data[_i_ker(i + 1, j, k)] + data[_i_ker(i - 1, j, k)]) / (hx * hx)
                        + (data[_i_ker(i, j + 1, k)] + data[_i_ker(i, j - 1, k)]) / (hy * hy)
                        + (data[_i_ker(i, j, k + 1)] + data[_i_ker(i, j, k - 1)]) / (hz * hz)
                    ) / divisor;
            }
        }
    }
}

__global__ void kernel_get_diffs(
    double *data, 
    double *next, 
    int block_size_x, 
    int block_size_y, 
    int block_size_z
) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;
    int offsetz = blockDim.z * gridDim.z;

    for (int i = idz - 1; i < block_size_x + 1; i += offsetz)
        for (int j = idy - 1; j < block_size_y + 1; j += offsety)
            for (int k = idx - 1; k < block_size_z + 1; k += offsetx) {

                bool out_of_bounds = (
                    i == -1 || j == -1 || k == -1 || i == block_size_x || j == block_size_y || k == block_size_z
                );

                if (out_of_bounds) {
                    data[_i_ker(i, j, k)] = 0.0;
                } else {
                    data[_i_ker(i, j, k)] = std::fabs(next[_i_ker(i, j, k)] - data[_i_ker(i, j, k)]);
                }
            }
}

int main(int argc, char *argv[]) {
    int device_cnt;
    cudaGetDeviceCount(&device_cnt);

    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    
    auto start_time = std::chrono::high_resolution_clock::now();

    char output_file[128];

    int ib, jb, kb;

    NBlocksPerDim n_blocks;
    int nbx, nby, nbz; 

    BlockSize block_size;
    int nx, ny, nz;

    int i, j, k;

    DimSize len;
    double lx, ly, lz; 

    BorderCondition border;
    double border_bottom, border_top, border_left, border_right, border_face, border_back;

    double initial_temperature;
    double eps, global_difference;

    int id, numproc, proc_name_len;
    char proc_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Get_processor_name(proc_name, &proc_name_len);

    cudaSetDevice(id % device_cnt);

    if (id == 0) {
        std::cin >> nbx >> nby >> nbz;
        n_blocks = NBlocksPerDim(nbx, nby, nbz);

        std::cin >> nx >> ny >> nz;
        block_size = BlockSize(nx, ny, nz);

        std::cin >> output_file;
        std::cin >> eps;

        std::cin >> lx >> ly >> lz;
        len = DimSize(lx, ly, lz);

        std::cin >> border_bottom >> border_top >> border_left >> border_right >> border_face >> border_back;
        border = BorderCondition(border_bottom, border_top, border_left, border_right, border_face, border_back);

        std::cin >> initial_temperature;
    }

    MPI_Bcast(&n_blocks, 3, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&block_size, 3, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&len, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&border, 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&initial_temperature, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&eps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(output_file, 128, MPI_CHAR, 0, MPI_COMM_WORLD);

    H h(len.x / (block_size.x * n_blocks.x), 
        len.y / (block_size.y * n_blocks.y), 
        len.z / (block_size.z * n_blocks.z));

    double *data, *next, *buff, *temp;
    int data_size = sizeof(double) * (block_size.x + 2) * (block_size.y + 2) * (block_size.z + 2);
    data = (double*)malloc(data_size);
    next = (double*)malloc(data_size);

    int buffer_size;
    int incount = std::max({block_size.x, block_size.y, block_size.z}) + 2;
    incount *= incount;
    MPI_Pack_size(incount, MPI_DOUBLE, MPI_COMM_WORLD, &buffer_size);
    buff = (double*)malloc(buffer_size);

    double *dev_data, *dev_next, *dev_buff;

    CSC(cudaMalloc(&dev_data, data_size));
    CSC(cudaMalloc(&dev_next, data_size));
    CSC(cudaMalloc(&dev_buff, buffer_size));

    for (i = 0; i < block_size.x; i++) {
        for (j = 0; j < block_size.y; j++) {
            for (k = 0; k < block_size.z; k++) {
                data[_i(i, j, k)] = initial_temperature;
            }
        }
    }

    CSC(cudaMemcpy(dev_data, data, data_size, cudaMemcpyHostToDevice));

    ib = _ibx(id);
    jb = _iby(id);
    kb = _ibz(id);

    dim3 blocks(32, 32);
    dim3 threads(32, 32);

    do {
        kernel_set_border<<<blocks, threads>>>(
            dev_data, block_size.x, block_size.y, block_size.z, border.left, border.right, 'x'
        );
        CSC(cudaGetLastError());

        kernel_set_border<<<blocks, threads>>>(
            dev_data, block_size.x, block_size.y, block_size.z, border.face, border.back, 'y'
        );
        CSC(cudaGetLastError());

        kernel_set_border<<<blocks, threads>>>(
            dev_data, block_size.x, block_size.y, block_size.z, border.bottom, border.top, 'z'
        );
        CSC(cudaGetLastError());

        if (ib + 1 < n_blocks.x) {
            kernel_fill_buff_yz<<<blocks, threads>>>(
                dev_data, dev_buff, block_size.x, block_size.y, block_size.z, block_size.x
            );
            CSC(cudaGetLastError());
            CSC(cudaMemcpy(buff, dev_buff, buffer_size, cudaMemcpyDeviceToHost));
            MPI_Send(buff, block_size.y * block_size.z, MPI_DOUBLE, 
                _ib(ib + 1, jb, kb), id, MPI_COMM_WORLD);
        }

        if (jb + 1 < n_blocks.y) {
            kernel_fill_buff_xz<<<blocks, threads>>>(
                dev_data, dev_buff, block_size.x, block_size.y, block_size.z, block_size.y
            );
            CSC(cudaGetLastError());
            CSC(cudaMemcpy(buff, dev_buff, buffer_size, cudaMemcpyDeviceToHost));
            MPI_Send(buff, block_size.x * block_size.z, MPI_DOUBLE,
                _ib(ib, jb + 1, kb), id, MPI_COMM_WORLD);
        }

        if (kb + 1 < n_blocks.z) {
            kernel_fill_buff_xy<<<blocks, threads>>>(
                dev_data, dev_buff, block_size.x, block_size.y, block_size.z, block_size.z
            );
            CSC(cudaGetLastError());
            CSC(cudaMemcpy(buff, dev_buff, buffer_size, cudaMemcpyDeviceToHost));
            MPI_Send(buff, block_size.x * block_size.y, MPI_DOUBLE, 
                _ib(ib, jb, kb + 1), id, MPI_COMM_WORLD);
        }

        if (ib > 0) {
            MPI_Recv(buff, block_size.y * block_size.z, MPI_DOUBLE, 
                _ib(ib - 1, jb, kb), _ib(ib - 1, jb, kb), MPI_COMM_WORLD, &status);
   
            CSC(cudaMemcpy(dev_buff, buff, buffer_size, cudaMemcpyHostToDevice));
            kernel_fill_data_yz<<<blocks, threads>>>(
                dev_data, dev_buff, block_size.x, block_size.y, block_size.z, -1
            );
            CSC(cudaGetLastError());
        }

        if (jb > 0) {
            MPI_Recv(buff, block_size.x * block_size.z, MPI_DOUBLE, 
                _ib(ib, jb - 1, kb), _ib(ib, jb - 1, kb), MPI_COMM_WORLD, &status);
                
            CSC(cudaMemcpy(dev_buff, buff, buffer_size, cudaMemcpyHostToDevice));
            kernel_fill_data_xz<<<blocks, threads>>>(
                dev_data, dev_buff, block_size.x, block_size.y, block_size.z, -1
            );
            CSC(cudaGetLastError());
        }

        if (kb > 0) {
            MPI_Recv(buff, block_size.x * block_size.y, MPI_DOUBLE, 
                _ib(ib, jb, kb - 1), _ib(ib, jb, kb - 1), MPI_COMM_WORLD, &status);

            CSC(cudaMemcpy(dev_buff, buff, buffer_size, cudaMemcpyHostToDevice));
            kernel_fill_data_xy<<<blocks, threads>>>(
                dev_data, dev_buff, block_size.x, block_size.y, block_size.z, -1
            );
            CSC(cudaGetLastError());
        }

        if (ib > 0) {
            kernel_fill_buff_yz<<<blocks, threads>>>(
                dev_data, dev_buff, block_size.x, block_size.y, block_size.z, 1
            );
            CSC(cudaGetLastError());
            CSC(cudaMemcpy(buff, dev_buff, buffer_size, cudaMemcpyDeviceToHost));
            
            MPI_Send(buff, block_size.y * block_size.z, MPI_DOUBLE, 
                _ib(ib - 1, jb, kb), id, MPI_COMM_WORLD);
        }

        if (jb > 0) {
            kernel_fill_buff_xz<<<blocks, threads>>>(
                dev_data, dev_buff, block_size.x, block_size.y, block_size.z, 1
            );
            CSC(cudaGetLastError());
            CSC(cudaMemcpy(buff, dev_buff, buffer_size, cudaMemcpyDeviceToHost));
            
            MPI_Send(buff, block_size.x * block_size.z, MPI_DOUBLE, 
                _ib(ib, jb - 1, kb), id, MPI_COMM_WORLD);
        }

        if (kb > 0) {
            kernel_fill_buff_xy<<<blocks, threads>>>(
                dev_data, dev_buff, block_size.x, block_size.y, block_size.z, 1
            );
            CSC(cudaGetLastError());
            CSC(cudaMemcpy(buff, dev_buff, buffer_size, cudaMemcpyDeviceToHost));
            
            MPI_Send(buff, block_size.x * block_size.y, MPI_DOUBLE, 
                _ib(ib, jb, kb - 1), id, MPI_COMM_WORLD);
        }

        if (ib + 1 < n_blocks.x) {
            MPI_Recv(buff, block_size.y * block_size.z, MPI_DOUBLE, 
                _ib(ib + 1, jb, kb), _ib(ib + 1, jb, kb), MPI_COMM_WORLD, &status);

            CSC(cudaMemcpy(dev_buff, buff, buffer_size, cudaMemcpyHostToDevice));
            kernel_fill_data_yz<<<blocks, threads>>>(
                dev_data, dev_buff, block_size.x, block_size.y, block_size.z, block_size.x
            );
            CSC(cudaGetLastError());
        }

        if (jb + 1 < n_blocks.y) {
            MPI_Recv(buff, block_size.x * block_size.z, MPI_DOUBLE, 
                _ib(ib, jb + 1, kb), _ib(ib, jb + 1, kb), MPI_COMM_WORLD, &status);
                
            CSC(cudaMemcpy(dev_buff, buff, buffer_size, cudaMemcpyHostToDevice));
            kernel_fill_data_xz<<<blocks, threads>>>(
                dev_data, dev_buff, block_size.x, block_size.y, block_size.z, block_size.y
            );
            CSC(cudaGetLastError());
        }

        if (kb + 1 < n_blocks.z) {
            MPI_Recv(buff, block_size.x * block_size.y, MPI_DOUBLE, 
                _ib(ib, jb, kb + 1), _ib(ib, jb, kb + 1), MPI_COMM_WORLD, &status);
                
            CSC(cudaMemcpy(dev_buff, buff, buffer_size, cudaMemcpyHostToDevice));
            kernel_fill_data_xy<<<blocks, threads>>>(
                dev_data, dev_buff, block_size.x, block_size.y, block_size.z, block_size.z
            );
            CSC(cudaGetLastError());
        }

        kernel_main_loop<<<dim3(4,4,4), dim3(8,8,8)>>>(
            dev_data, dev_next, block_size.x, block_size.y, block_size.z, h.x, h.y, h.z
        );
        CSC(cudaGetLastError());

        kernel_get_diffs<<<dim3(4,4,4), dim3(8,8,8)>>>(
            dev_data, dev_next, block_size.x, block_size.y, block_size.z
        );
        CSC(cudaGetLastError());

        double max_diff = 0.0;
        thrust::device_ptr<double> diffs_p = thrust::device_pointer_cast(dev_data);
        thrust::device_ptr<double> max_diff_p = thrust::max_element(diffs_p, diffs_p + (block_size.x + 2) * (block_size.y + 2) * (block_size.z + 2));
        max_diff = *max_diff_p;

        MPI_Allreduce(&max_diff, &global_difference, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        temp = dev_next;
        dev_next = dev_data;
        dev_data = temp;
        // global_difference = 0;
        std::cout << global_difference << std::endl;
    } while (global_difference >= eps);

    CSC(cudaMemcpy(data, dev_data, data_size, cudaMemcpyDeviceToHost));

    // отводим на представление каждого числа n_size символов
    // int n_size = 14;
    int n_size = 20;
    int buff_file_size = block_size.x * block_size.y * block_size.z * n_size;
    char* buff_file = (char*)malloc(buff_file_size);
    memset(buff_file, ' ', buff_file_size);

    for (k = 0; k < block_size.z; ++k) {
        for (j = 0; j < block_size.y; ++j) {
            for (i = 0; i < block_size.x; ++i) {
                int offset = (k * block_size.x * block_size.y + j * block_size.x + i) * n_size;
                // sprintf(buff_file + offset, "%.7e", data[_i(i, j, k)]);
                sprintf(buff_file + offset, "%.6e", data[_i(i, j, k)]);
            }
        }
    }

    for (i = 0; i < buff_file_size; ++i) {
        if (buff_file[i] == '\0') {
            buff_file[i] = ' ';
        }
    }

    MPI_File fp;
    MPI_Datatype type1;
    MPI_Datatype type2;
    MPI_Aint stride1 = n_blocks.x * block_size.x * n_size;
    MPI_Aint stride2 = n_blocks.y * block_size.y * stride1;

    MPI_Type_create_hvector(block_size.y,
                            block_size.x * n_size,
                            stride1,
                            MPI_CHAR,
                            &type1);
    MPI_Type_commit(&type1);

    MPI_Type_create_hvector(block_size.z,
                            1,
                            stride2,
                            type1,
                            &type2);
    MPI_Type_commit(&type2);
    
    MPI_File_delete(output_file, MPI_INFO_NULL);
    MPI_File_open(MPI_COMM_WORLD, output_file, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fp);

    MPI_Offset disp = ib * block_size.x * n_size + jb * stride1 * block_size.y + kb * block_size.z * stride2;
    MPI_File_set_view(fp, disp, MPI_CHAR, type2, "native", MPI_INFO_NULL);
    MPI_File_write_all(
        fp, buff_file, block_size.x * block_size.y * block_size.z * n_size, MPI_CHAR, MPI_STATUS_IGNORE
    );
    MPI_File_close(&fp);

    MPI_Finalize();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto time = end_time - start_time;

    std::cout << "took " << time/std::chrono::milliseconds(1) << "ms to run.\n";
    
    free(data);
    free(next);
    free(buff);
    free(buff_file);
    CSC(cudaFree(dev_data));
    CSC(cudaFree(dev_next));
    CSC(cudaFree(dev_buff));

    return 0;
}