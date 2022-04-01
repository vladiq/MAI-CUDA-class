#include <iostream>
#include <string>

#define CSC(call)  													\
do {																\
	cudaError_t res = call;											\
	if (res != cudaSuccess) {										\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);													\
	}																\
} while(0)

texture<uchar4, 2, cudaReadModeElementType> tex;

__global__ void ssaa_kernel(
	uchar4* dev_out, int w_new, int h_new, int w_stride, int h_stride
) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;

    int filter_size = w_stride * h_stride;
    int r, g, b, alpha = 0;
    uchar4 p;

    for(int y = idy; y < h_new; y += offsety)
		for(int x = idx; x < w_new; x += offsetx) {
            r = 0;
            g = 0;
            b = 0;
			alpha = 0;
            for (int i = 0; i < w_stride; ++i) {
                for (int j = 0; j < h_stride; ++j) {
                    p = tex2D(tex, x * w_stride + i, y * h_stride + j);
                    r += p.x;
                    g += p.y;
                    b += p.z;
					alpha += p.w;
                }
            }
            r /= filter_size;
            g /= filter_size;
            b /= filter_size;
            dev_out[y * w_new + x] = make_uchar4(r, g, b, alpha);
		}
}


int main() {
    std::string in_file, out_file;
    int w_new, h_new;
    std::cin >> in_file >> out_file >> w_new >> h_new;

    int w, h;
    FILE *fp = fopen(in_file.c_str(), "rb");
	fread(&w, sizeof(int), 1, fp);
	fread(&h, sizeof(int), 1, fp);
    uchar4* data = new uchar4[w * h];
	fread(data, sizeof(uchar4), w * h, fp);
	fclose(fp);

		cudaArray *arr;
		cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
		CSC(cudaMallocArray(&arr, &ch, w, h));
		CSC(cudaMemcpyToArray(
			arr, 0, 0, data, sizeof(uchar4) * w * h, cudaMemcpyHostToDevice
		));
		tex.addressMode[0] = cudaAddressModeClamp;
		tex.addressMode[1] = cudaAddressModeClamp;
		tex.channelDesc = ch;
		tex.filterMode = cudaFilterModePoint;
		tex.normalized = false;
		CSC(cudaBindTextureToArray(tex, arr, ch));

		uchar4* dev_out;
		CSC(cudaMalloc(&dev_out, sizeof(uchar4) * w_new * h_new));
		int w_stride = w / w_new;
		int h_stride = h / h_new;
		ssaa_kernel<<<dim3(16, 32), dim3(16, 32)>>>(dev_out, w_new, h_new, w_stride, h_stride);
		CSC(cudaGetLastError());
		CSC(cudaMemcpy(data, dev_out, sizeof(uchar4) * w_new * h_new, cudaMemcpyDeviceToHost));

		CSC(cudaUnbindTexture(tex));
		CSC(cudaFreeArray(arr));
		CSC(cudaFree(dev_out));

		fp = fopen(out_file.c_str(), "wb");
		fwrite(&w_new, sizeof(int), 1, fp);
		fwrite(&h_new, sizeof(int), 1, fp);
		fwrite(data, sizeof(uchar4), w_new * h_new, fp);
		fclose(fp);

	delete[] data;
}