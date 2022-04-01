#include <iostream>
#include <string>
#include <vector>
#include <float.h>

#define CSC(call)  													\
do {																\
	cudaError_t res = call;											\
	if (res != cudaSuccess) {										\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);													\
	}																\
} while(0)

const int MAX_N_CLASSES = 32;

__constant__ double mean_dev[MAX_N_CLASSES][3];
__constant__ double cov_dev[MAX_N_CLASSES][3][3];
__constant__ double det_dev[MAX_N_CLASSES];
__constant__ double cov_inv_dev[MAX_N_CLASSES][3][3];

__global__ void classifier(uchar4* data, int w, int h, int n_classes) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;
    uchar4 pixel;

    for (int y = idy; y < h; y += offsety) {
        for (int x = idx; x < w; x += offsetx) {
            pixel = data[y * w + x];

            // init max statistics
            double max_likelihood = -DBL_MAX;
            int max_class_idx = -1;

            // get likelihoods of each class for current pixel p
            for (int i = 0; i < n_classes; ++i) {

                double diffs[3];
                diffs[0] = pixel.x - mean_dev[i][0];
                diffs[1] = pixel.y - mean_dev[i][1];
                diffs[2] = pixel.z - mean_dev[i][2];

                double diff_mult_cov[3];
                for (int j = 0; j < 3; ++j) {
                    diff_mult_cov[j] = 0;
                    for (int k = 0; k < 3; ++k) {
                        // row_vector-matrix multiplication
                        diff_mult_cov[j] += diffs[k] * cov_inv_dev[i][k][j];
                    }
                }

                double mle = 0;
                for (int j = 0; j < 3; ++j) {
                    mle += diff_mult_cov[j] * diffs[j];
                }
                mle = - mle - std::log(std::abs(det_dev[i]));

                if (mle > max_likelihood) {
                    max_likelihood = mle;
                    max_class_idx = i;
                }
            }
    
            // set the alpha channel with chosen class index
            data[y * w + x].w = max_class_idx;
        }
    }
}


int main() {
    std::string in_file, out_file;
    int n_classes;
    std::cin >> in_file >> out_file >> n_classes;
    std::vector<std::vector<std::pair<int, int>>> classes(n_classes);
    for (int class_idx = 0; class_idx < n_classes; ++class_idx) {
        int n_points;
        std::cin >> n_points;
        classes[class_idx].resize(n_points);
        for (int point_idx = 0; point_idx < n_points; ++point_idx) {
            std::cin >> classes[class_idx][point_idx].first >> classes[class_idx][point_idx].second;
        }
    }

    int w, h;
    FILE *fp = fopen(in_file.c_str(), "rb");
	fread(&w, sizeof(int), 1, fp);
	fread(&h, sizeof(int), 1, fp);
    uchar4* data = new uchar4[w * h];
	fread(data, sizeof(uchar4), w * h, fp);
	fclose(fp);

    // compute memle
    double mean[MAX_N_CLASSES][3] = {0};

    for (int i = 0; i < n_classes; ++i) {
        int n_points = classes[i].size();

        for (int j = 0; j < n_points; ++j) {
            uchar4 cur_pixel = data[classes[i][j].first + classes[i][j].second * w];
            
            // RGB channels
            mean[i][0] += cur_pixel.x;
            mean[i][1] += cur_pixel.y;
            mean[i][2] += cur_pixel.z;
        }

        // normalize by number of training points
        mean[i][0] /= n_points;
        mean[i][1] /= n_points;
        mean[i][2] /= n_points;
    }


    // compute covariance matrix
    double cov[MAX_N_CLASSES][3][3] = {0};
    for (int i = 0; i < n_classes; ++i) {
        int n_points = classes[i].size();

        for (int j = 0; j < n_points; ++j) {
            std::vector<double> diffs(3, 0);

            uchar4 cur_pixel = data[classes[i][j].first + classes[i][j].second * w];
            std::vector<unsigned char> cur_rgb_values = {cur_pixel.x, cur_pixel.y, cur_pixel.z};

            for (int channel = 0; channel < 3; ++channel) {
                diffs[channel] = static_cast<double>(cur_rgb_values[channel]) - mean[i][channel];
            }

            for (int c1 = 0; c1 < 3; ++c1) {
                for (int c2 = 0; c2 < 3; ++c2) {
                    cov[i][c1][c2] += diffs[c1] * diffs[c2];
                }
            }
        }

        // normalize by number of points
        for (int c1 = 0; c1 < 3; ++c1) {
            for (int c2 = 0; c2 < 3; ++c2) {
                cov[i][c1][c2] /= n_points - 1;
            }
        }
    }

    double determinant[MAX_N_CLASSES] = {0};
    double cov_inverse[MAX_N_CLASSES][3][3];
    for (int i = 0; i < n_classes; ++i) {
        determinant[i] = cov[i][0][0] * (cov[i][1][1] * cov[i][2][2] - cov[i][2][1] * cov[i][1][2]) -
                        cov[i][0][1] * (cov[i][1][0] * cov[i][2][2] - cov[i][2][0] * cov[i][1][2]) +
                        cov[i][0][2] * (cov[i][1][0] * cov[i][2][1] - cov[i][2][0] * cov[i][1][1]);

        cov_inverse[i][0][0] =  (cov[i][1][1] * cov[i][2][2] - cov[i][2][1] * cov[i][1][2]) / determinant[i];
        cov_inverse[i][1][0] = -(cov[i][1][0] * cov[i][2][2] - cov[i][2][0] * cov[i][1][2]) / determinant[i];
        cov_inverse[i][2][0] =  (cov[i][1][0] * cov[i][2][1] - cov[i][2][0] * cov[i][1][1]) / determinant[i];
        cov_inverse[i][0][1] = -(cov[i][0][1] * cov[i][2][2] - cov[i][2][1] * cov[i][0][2]) / determinant[i];
        cov_inverse[i][1][1] =  (cov[i][0][0] * cov[i][2][2] - cov[i][2][0] * cov[i][0][2]) / determinant[i];
        cov_inverse[i][2][1] = -(cov[i][0][0] * cov[i][2][1] - cov[i][2][0] * cov[i][0][1]) / determinant[i];
        cov_inverse[i][0][2] =  (cov[i][0][1] * cov[i][1][2] - cov[i][1][1] * cov[i][0][2]) / determinant[i];
        cov_inverse[i][1][2] = -(cov[i][0][0] * cov[i][1][2] - cov[i][1][0] * cov[i][0][2]) / determinant[i];
        cov_inverse[i][2][2] =  (cov[i][0][0] * cov[i][1][1] - cov[i][1][0] * cov[i][0][1]) / determinant[i];
    }

    CSC(cudaMemcpyToSymbol(mean_dev, mean, sizeof(double) * MAX_N_CLASSES * 3));
    CSC(cudaMemcpyToSymbol(cov_dev, cov, sizeof(double) * MAX_N_CLASSES * 3 * 3));
    CSC(cudaMemcpyToSymbol(cov_inv_dev, cov_inverse, sizeof(double) * MAX_N_CLASSES * 3 * 3));
    CSC(cudaMemcpyToSymbol(det_dev, determinant, sizeof(double) * MAX_N_CLASSES));
    
    uchar4* data_dev;
    CSC(cudaMalloc(&data_dev, sizeof(uchar4) * h * w));
    CSC(cudaMemcpy(data_dev, data, sizeof(uchar4) * h * w, cudaMemcpyHostToDevice));
    classifier<<<dim3(32, 32), dim3(32, 32)>>>(data_dev, h, w, n_classes);
    CSC(cudaGetLastError());
    CSC(cudaMemcpy(data, data_dev, sizeof(uchar4) * h * w, cudaMemcpyDeviceToHost));

    fp = fopen(out_file.c_str(), "wb");
    fwrite(&w, sizeof(int), 1, fp);
    fwrite(&h, sizeof(int), 1, fp);
    fwrite(data, sizeof(uchar4), w * h, fp);
    fclose(fp);

    CSC(cudaFree(data_dev));
	delete[] data;
}