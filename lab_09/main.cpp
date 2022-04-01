#include <iostream>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>
#include "mpi.h"
#include <omp.h>

#define _i(i, j, k) (((i) + 1) +                                         \
                    (block_size.x + 2) * ((j) + 1) +                     \
                    (block_size.x + 2) * (block_size.y + 2) * ((k) + 1))

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

int main(int argc, char *argv[]) {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

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

    auto start_time = std::chrono::high_resolution_clock::now();

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

    double divisor = 2 * (1.0 / (h.x * h.x) + 1.0 / (h.y * h.y) + 1.0 / (h.z * h.z));
    
    double *data, *next, *temp;
    data = (double*)malloc(sizeof(double) * (block_size.x + 2) * (block_size.y + 2) * (block_size.z + 2));
    next = (double*)malloc(sizeof(double) * (block_size.x + 2) * (block_size.y + 2) * (block_size.z + 2));

    #pragma omp parallel for private(i, j, k) shared(data)
    for (i = 0; i < block_size.x; i++) {
        for (j = 0; j < block_size.y; j++) {
            for (k = 0; k < block_size.z; k++) {
                data[_i(i, j, k)] = initial_temperature;
            }
        }
    }

    ib = _ibx(id);
    jb = _iby(id);
    kb = _ibz(id);

    MPI_Aint stride;
    MPI_Datatype gather_x_xy;
    stride = sizeof(double) * (block_size.x + 2);
    MPI_Type_create_hvector(block_size.y, 1, stride, MPI_DOUBLE, &gather_x_xy);
    MPI_Type_commit(&gather_x_xy);

    MPI_Datatype gather_x_xyz;
    stride = sizeof(double) * (block_size.x + 2) * (block_size.y + 2);
	MPI_Type_create_hvector(block_size.z, 1, stride, gather_x_xy, &gather_x_xyz);
	MPI_Type_commit(&gather_x_xyz);

	MPI_Datatype gather_y_xyz;
    stride = sizeof(double) * (block_size.x + 2) * (block_size.y + 2);
	MPI_Type_create_hvector(block_size.z, block_size.x, stride, MPI_DOUBLE, &gather_y_xyz);
	MPI_Type_commit(&gather_y_xyz);

	MPI_Datatype gather_z_xyz;
    stride = sizeof(double) * (block_size.x + 2);
	MPI_Type_create_hvector(block_size.y, block_size.x, stride, MPI_DOUBLE, &gather_z_xyz);
	MPI_Type_commit(&gather_z_xyz);

    #pragma omp parallel for private(j, k) shared(data, next)
    for (j = 0; j < block_size.y; ++j) {
        for (k = 0; k < block_size.z; ++k) {
            int left_idx = _i(-1, j, k), right_idx = _i(block_size.x, j, k);
            data[left_idx] = border.left;
            next[left_idx] = border.left;
            data[right_idx] = border.right;
            next[right_idx] = border.right;
        }
    }

    #pragma omp parallel for private(i, k) shared(data, next)
    for (i = 0; i < block_size.x; ++i) {
        for (k = 0; k < block_size.z; ++k) {
            int face_idx = _i(i, -1, k), back_idx = _i(i, block_size.y, k);
            data[face_idx] = border.face;
            next[face_idx] = border.face;
            data[back_idx] = border.back;
            next[back_idx] = border.back;
        }
    }

    #pragma omp parallel for private(i, j) shared(data, next)
    for (i = 0; i < block_size.x; ++i) {
        for (j = 0; j < block_size.y; ++j) {
            int bottom_idx = _i(i, j, -1), top_idx = _i(i, j, block_size.z);
            data[bottom_idx] = border.bottom;
            next[bottom_idx] = border.bottom;
            data[top_idx] = border.top;
            next[top_idx] = border.top;
        }
    }

    do {
        if (ib + 1 < n_blocks.x) {
            MPI_Send(&data[_i(block_size.x - 1, 0, 0)], 1, gather_x_xyz, 
                _ib(ib + 1, jb, kb), id, MPI_COMM_WORLD);
        }

        if (jb + 1 < n_blocks.y) {
            MPI_Send(&data[_i(0, block_size.y - 1, 0)], 1, gather_y_xyz,
                _ib(ib, jb + 1, kb), id, MPI_COMM_WORLD);
        }

        if (kb + 1 < n_blocks.z) {
            MPI_Send(&data[_i(0, 0, block_size.z - 1)], 1, gather_z_xyz, 
                _ib(ib, jb, kb + 1), id, MPI_COMM_WORLD);
        }

        if (ib > 0) {
            MPI_Recv(&data[_i(-1, 0, 0)], 1, gather_x_xyz, 
                _ib(ib - 1, jb, kb), _ib(ib - 1, jb, kb), MPI_COMM_WORLD, &status);
        }

        if (jb > 0) {
            MPI_Recv(&data[_i(0, -1, 0)], 1, gather_y_xyz, 
                _ib(ib, jb - 1, kb), _ib(ib, jb - 1, kb), MPI_COMM_WORLD, &status);
        }

        if (kb > 0) {
            MPI_Recv(&data[_i(0, 0, -1)], 1, gather_z_xyz, 
                _ib(ib, jb, kb - 1), _ib(ib, jb, kb - 1), MPI_COMM_WORLD, &status);
        }

        if (ib > 0) {
            MPI_Send(&data[_i(0, 0, 0)], 1, gather_x_xyz, 
                _ib(ib - 1, jb, kb), id, MPI_COMM_WORLD);
        }

        if (jb > 0) {
            MPI_Send(&data[_i(0, 0, 0)], 1, gather_y_xyz, 
                _ib(ib, jb - 1, kb), id, MPI_COMM_WORLD);
        }

        if (kb > 0) {
            MPI_Send(&data[_i(0, 0, 0)], 1, gather_z_xyz, 
                _ib(ib, jb, kb - 1), id, MPI_COMM_WORLD);
        }

        if (ib + 1 < n_blocks.x) {
            MPI_Recv(&data[_i(block_size.x, 0, 0)], 1, gather_x_xyz, 
                _ib(ib + 1, jb, kb), _ib(ib + 1, jb, kb), MPI_COMM_WORLD, &status);
        }

        if (jb + 1 < n_blocks.y) {
            MPI_Recv(&data[_i(0, block_size.y, 0)], 1, gather_y_xyz, 
                _ib(ib, jb + 1, kb), _ib(ib, jb + 1, kb), MPI_COMM_WORLD, &status);
        }

        if (kb + 1 < n_blocks.z) {
            MPI_Recv(&data[_i(0, 0, block_size.z)], 1, gather_z_xyz, 
                _ib(ib, jb, kb + 1), _ib(ib, jb, kb + 1), MPI_COMM_WORLD, &status);
        }

        double max_diff = 0.0;

	    #pragma omp parallel for private(i, j, k) shared(data, next, divisor, block_size, h) reduction(max : max_diff)
        for (i = 0; i < block_size.x; ++i) {
            for (j = 0; j < block_size.y; ++j) {
                for (k = 0; k < block_size.z; ++k) {
                    next[_i(i, j, k)] = (
                        (data[_i(i + 1, j, k)] + data[_i(i - 1, j, k)]) / (h.x * h.x)
                      + (data[_i(i, j + 1, k)] + data[_i(i, j - 1, k)]) / (h.y * h.y)
                      + (data[_i(i, j, k + 1)] + data[_i(i, j, k - 1)]) / (h.z * h.z)
                    ) / divisor;
                    max_diff = std::max(
                        max_diff, 
                        std::abs(next[_i(i, j, k)] - data[_i(i, j, k)])
                    );
                }
            }
        }

        MPI_Allreduce(
            &max_diff, &global_difference, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD
        );

        temp = next;
        next = data;
        data = temp;
        
        if (id == 0) {
            std::cerr << global_difference << std::endl;
        }
    } while (global_difference > eps);

    int n_size = 15;
    // int n_size = 20;
    size_t buff_file_size = block_size.x * block_size.y * block_size.z * n_size;
    char* buff_file = (char*)malloc(buff_file_size);
    memset(buff_file, ' ', buff_file_size);

    for (k = 0; k < block_size.z; ++k) {
        for (j = 0; j < block_size.y; ++j) {
            for (i = 0; i < block_size.x; ++i) {
                int offset = (k * block_size.x * block_size.y + j * block_size.x + i) * n_size;
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
    std::cerr << "done_write_all ";

    MPI_File_close(&fp);

    MPI_Finalize();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto time = end_time - start_time;
    std::cout << "took " << time/std::chrono::milliseconds(1) << "ms to run.\n";

    free(data);
    free(next);
    free(buff_file);

    return 0;
}