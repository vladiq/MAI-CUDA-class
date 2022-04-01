#include <iostream>
#include <string>
#include <cmath>
#include <algorithm>
#include <chrono>
#include "mpi/mpi.h"

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

    auto start_time = std::chrono::high_resolution_clock::now();

    std::string output_file;

    // трёхмерная сетка
    int ib, jb, kb;

    // кол-во блоков по измерениям
    NBlocksPerDim n_blocks;
    int nbx, nby, nbz; 

    // размеры каждого блока
    BlockSize block_size;
    int nx, ny, nz;

    // итерации внутри циклов
    int i, j, k;

    // размеры каждой области
    DimSize len;
    double lx, ly, lz;

    // граничные условия
    BorderCondition border;
    double border_bottom, border_top, border_left, border_right, border_face, border_back; // d, u, l, r, f, b;

    double initial_temperature;
    double eps, global_difference;

    int id, numproc, proc_name_len;
    char proc_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Get_processor_name(proc_name, &proc_name_len);

    // в нулевом процессе считываем данные
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

    H h(len.x / (block_size.x * n_blocks.x), 
        len.y / (block_size.y * n_blocks.y), 
        len.z / (block_size.z * n_blocks.z));

    // массивы для хранения, свапа и передачи данных
    double *data, *next, *buff, *temp;
    data = (double*)malloc(sizeof(double) * (block_size.x + 2) * (block_size.y + 2) * (block_size.z + 2));
    next = (double*)malloc(sizeof(double) * (block_size.x + 2) * (block_size.y + 2) * (block_size.z + 2));

    int buffer_size;
    int incount = std::max({block_size.x, block_size.y, block_size.z}) + 2;
    incount *= incount;
    MPI_Pack_size(incount, MPI_DOUBLE, MPI_COMM_WORLD, &buffer_size);
    buff = (double*)malloc(buffer_size);
    // MPI_Buffer_attach(buff, buffer_size);

    for (i = 0; i < block_size.x; i++) {
        for (j = 0; j < block_size.y; j++) {
            for (k = 0; k < block_size.z; k++) {
                data[_i(i, j, k)] = initial_temperature;
            }
        }
    }

    // переходим от одномерной индексации к трёхмерной
    ib = _ibx(id);
    jb = _iby(id);
    kb = _ibz(id);

    do {
        for (j = 0; j < block_size.y; ++j) {
            for (k = 0; k < block_size.z; ++k) {
                data[_i(-1, j, k)] = border.left;
                data[_i(block_size.x, j, k)] = border.right;
            }
        }

        for (i = 0; i < block_size.x; ++i) {
            for (k = 0; k < block_size.z; ++k) {
                data[_i(i, -1, k)] = border.face;
                data[_i(i, block_size.y, k)] = border.back;
            }
        }

        for (i = 0; i < block_size.x; ++i) {
            for (j = 0; j < block_size.y; ++j) {
                data[_i(i, j, -1)] = border.bottom;
                data[_i(i, j, block_size.z)] = border.top;
            }
        }

        if (ib + 1 < n_blocks.x) {
            for (j = 0; j < block_size.y; ++j) {
                for (k = 0; k < block_size.z; ++k) {
                    buff[k * block_size.y + j] = data[_i(block_size.x - 1, j, k)];
                }
            }
            MPI_Send(buff, block_size.y * block_size.z, MPI_DOUBLE, 
                _ib(ib + 1, jb, kb), id, MPI_COMM_WORLD);
        }

        if (jb + 1 < n_blocks.y) {
            for (i = 0; i < block_size.x; ++i) {
                for (k = 0; k < block_size.z; ++k) {
                    buff[k * block_size.x + i] = data[_i(i, block_size.y - 1, k)];
                }
            }
            MPI_Send(buff, block_size.x * block_size.z, MPI_DOUBLE,
                _ib(ib, jb + 1, kb), id, MPI_COMM_WORLD);
        }

        if (kb + 1 < n_blocks.z) {
            for (i = 0; i < block_size.x; ++i) {
                for (j = 0; j < block_size.y; ++j) {
                    buff[j * block_size.x + i] = data[_i(i, j, block_size.z - 1)];
                }
            }
            MPI_Send(buff, block_size.x * block_size.y, MPI_DOUBLE, 
                _ib(ib, jb, kb + 1), id, MPI_COMM_WORLD);
        }

        if (ib > 0) {
            MPI_Recv(buff, block_size.y * block_size.z, MPI_DOUBLE, 
                _ib(ib - 1, jb, kb), _ib(ib - 1, jb, kb), MPI_COMM_WORLD, &status);
   
            for (j = 0; j < block_size.y; ++j) {
                for (k = 0; k < block_size.z; ++k) {
                    data[_i(-1, j, k)] = buff[k * block_size.y + j];
                }
            }
        }

        if (jb > 0) {
            MPI_Recv(buff, block_size.x * block_size.z, MPI_DOUBLE, 
                _ib(ib, jb - 1, kb), _ib(ib, jb - 1, kb), MPI_COMM_WORLD, &status);
                
            for (i = 0; i < block_size.x; ++i) {
                for (k = 0; k < block_size.z; ++k) {
                    data[_i(i, -1, k)] = buff[k * block_size.x + i];
                }
            }
        }

        if (kb > 0) {
            MPI_Recv(buff, block_size.x * block_size.y, MPI_DOUBLE, 
                _ib(ib, jb, kb - 1), _ib(ib, jb, kb - 1), MPI_COMM_WORLD, &status);
                
            for (i = 0; i < block_size.x; ++i) {
                for (j = 0; j < block_size.y; ++j) {
                    data[_i(i, j, -1)] = buff[j * block_size.x + i];
                }
            }
        }

        if (ib > 0) {
            for (j = 0; j < block_size.y; ++j) {
                for (k = 0; k < block_size.z; ++k) {
                    buff[k * block_size.y + j] = data[_i(0, j, k)];
                }
            }
            MPI_Send(buff, block_size.y * block_size.z, MPI_DOUBLE, 
                _ib(ib - 1, jb, kb), id, MPI_COMM_WORLD);
        }

        if (jb > 0) {
            for (i = 0; i < block_size.x; ++i) {
                for (k = 0; k < block_size.z; ++k) {
                    buff[k * block_size.x + i] = data[_i(i, 0, k)];
                }
            }
            MPI_Send(buff, block_size.x * block_size.z, MPI_DOUBLE, 
                _ib(ib, jb - 1, kb), id, MPI_COMM_WORLD);
        }

        if (kb > 0) {
            for (i = 0; i < block_size.x; ++i) {
                for (j = 0; j < block_size.y; ++j) {
                    buff[j * block_size.x + i] = data[_i(i, j, 0)];
                }
            }
            MPI_Send(buff, block_size.x * block_size.y, MPI_DOUBLE, 
                _ib(ib, jb, kb - 1), id, MPI_COMM_WORLD);
        }

        if (ib + 1 < n_blocks.x) {
            MPI_Recv(buff, block_size.y * block_size.z, MPI_DOUBLE, 
                _ib(ib + 1, jb, kb), _ib(ib + 1, jb, kb), MPI_COMM_WORLD, &status);

            for (j = 0; j < block_size.y; ++j) {
                for (k = 0; k < block_size.z; ++k) {
                    data[_i(block_size.x, j, k)] = buff[k * block_size.y + j];
                }
            }
        }

        if (jb + 1 < n_blocks.y) {
            MPI_Recv(buff, block_size.x * block_size.z, MPI_DOUBLE, 
                _ib(ib, jb + 1, kb), _ib(ib, jb + 1, kb), MPI_COMM_WORLD, &status);
                
            for (i = 0; i < block_size.x; ++i) {
                for (k = 0; k < block_size.z; ++k) {
                    data[_i(i, block_size.y, k)] = buff[k * block_size.x + i];
                }
            }
        }

        if (kb + 1 < n_blocks.z) {
            MPI_Recv(buff, block_size.x * block_size.y, MPI_DOUBLE, 
                _ib(ib, jb, kb + 1), _ib(ib, jb, kb + 1), MPI_COMM_WORLD, &status);
                
            for (i = 0; i < block_size.x; ++i) {
                for (j = 0; j < block_size.y; ++j) {
                    data[_i(i, j, block_size.z)] = buff[j * block_size.x + i];
                }
            }
        }

        double max_diff = 0.0;
        double divisor = 2 * (1.0 / (h.x * h.x) + 1.0 / (h.y * h.y) + 1.0 / (h.z * h.z));

        for (i = 0; i < block_size.x; ++i) {
            for (j = 0; j < block_size.y; ++j) {
                for (k = 0; k < block_size.z; ++k) {
                    next[_i(i, j, k)] = (
                        (data[_i(i + 1, j, k)] + data[_i(i - 1, j, k)]) / (h.x * h.x)
                      + (data[_i(i, j + 1, k)] + data[_i(i, j - 1, k)]) / (h.y * h.y)
                      + (data[_i(i, j, k + 1)] + data[_i(i, j, k - 1)]) / (h.z * h.z)
                    ) / divisor;
                    double tmp = std::abs(next[_i(i, j, k)] - data[_i(i, j, k)]);
                    max_diff = std::max({max_diff, tmp});
                }
            }
        }

        MPI_Allreduce(&max_diff, &global_difference, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        temp = next;
        next = data;
        data = temp;
        std::cout << global_difference << std::endl;
    } while (global_difference > eps);

    if (id != 0) {
        for (k = 0; k < block_size.z; ++k) {
            for (j = 0; j < block_size.y; ++j) {
                for (i = 0; i < block_size.x; ++i) {
                    buff[i] = data[_i(i, j, k)];
                }
                MPI_Send(buff, block_size.x, MPI_DOUBLE, 0, id, MPI_COMM_WORLD);
            }
        }
    } else {
        FILE* file = std::fopen(output_file.c_str(), "w");
        for (kb = 0; kb < n_blocks.z; ++kb) {
            for (k = 0; k < block_size.z; ++k) {
                for (jb = 0; jb < n_blocks.y; ++jb) {
                    for (j = 0; j < block_size.y; ++j) {
                        for (ib = 0; ib < n_blocks.x; ++ib) {
                            if (_ib(ib, jb, kb) == 0) {
                                for (i = 0; i < block_size.x; ++i) {
                                    buff[i] = data[_i(i, j, k)];
                                }
                            } else {
                                MPI_Recv(
                                    buff, 
                                    block_size.x, 
                                    MPI_DOUBLE,
                                    _ib(ib, jb, kb),
                                    _ib(ib, jb, kb), 
                                    MPI_COMM_WORLD, 
                                    &status
                                );
                            }

                            for (i = 0; i < block_size.x; ++i) {
                                std::fprintf(file, "%.7e ", buff[i]);
                            }
                        }
                    }
                }
            }
        }
        std::fclose(file);
    }

    MPI_Finalize();

    free(data);
    free(next);
    free(buff);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto time = end_time - start_time;

    std::cout << "took " << time/std::chrono::milliseconds(1) << "ms to run.\n";

    return 0;
}