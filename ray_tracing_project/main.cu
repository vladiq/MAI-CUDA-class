#include <iostream>
#include <vector>
#include <string>
#include <chrono>

#define CSC(call)                                                   \
do {                                                                \
    cudaError_t X = call;                                           \
    if (X != cudaSuccess) {                                         \
        fprintf(stderr, "ERROR: in %s:%d. Message: %s\n",           \
                __FILE__, __LINE__, cudaGetErrorString(X));         \
        exit(0);                                                    \
    }                                                               \
} while(0)


struct polygon {
    float3 x, y, z;
    uchar4 color;
};

__host__ polygon get_polygon(
    int x, int y, int z, 
    const std::vector<float3> &verts, 
    uchar4 color
) {
    return {verts[x], verts[y], verts[z], color};
}

__host__ uchar4 color_to_uchar4(float3 color) {
    return make_uchar4(
        255 * color.x, 255 * color.y, 255 * color.z, 0
    );
}

__device__ __host__ float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __host__ float3 prod(float3 a, float3 b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__device__ __host__ float3 norm(float3 v) {
    float vectorlength = std::sqrt(dot(v, v));
    return make_float3(
        v.x / vectorlength,
        v.y / vectorlength,
        v.z / vectorlength
    );
}

__device__ __host__ float3 diff(float3 a, float3 b) {
    return make_float3(
        a.x - b.x,
        a.y - b.y,
        a.z - b.z
    );
}

__device__ __host__ float3 add(float3 a, float3 b) {
    return make_float3(
        a.x + b.x,
        a.y + b.y,
        a.z + b.z
    );
}

__device__ __host__ float3 mult(float3 a, float3 b, float3 c, float3 v) {
    return make_float3(
        a.x * v.x + b.x * v.y + c.x * v.z,
        a.y * v.x + b.y * v.y + c.y * v.z,
        a.z * v.x + b.z * v.y + c.z * v.z
    );
}


__host__ std::pair<float3, float3> get_camera_settings(
    float current_step,
    float r_0_c, float z_0_c, float phi_0_c,
    float A_r_c, float A_z_c,
    float omega_r_c, float omega_z_c, float omega_phi_c,
    float p_r_c, float p_z_c,
    float r_0_n, float z_0_n, float phi_0_n,
    float A_r_n, float A_z_n,
    float omega_r_n, float omega_z_n, float omega_phi_n,
    float p_r_n, float p_z_n
) {

    float r_c_t = r_0_c + A_r_c * std::sin(omega_r_c * current_step + p_r_c);
    float z_c_t = z_0_c + A_z_c * std::sin(omega_z_c * current_step + p_z_c);
    float phi_c_t = phi_0_c + omega_phi_c * current_step;
    float3 camera_pos = make_float3(
        std::cos(phi_c_t) * r_c_t, 
        std::sin(phi_c_t) * r_c_t,
        z_c_t
    );

    float r_n_t = r_0_n + A_r_n * std::sin(omega_r_n * current_step + p_r_n);
	float z_n_t = z_0_n + A_z_n * std::sin(omega_z_n * current_step + p_z_n);
	float phi_n_t = phi_0_n + omega_phi_n * current_step;
	float3 camera_viewpoint = make_float3(
        std::cos(phi_n_t) * r_n_t,
        std::sin(phi_n_t) * r_n_t, 
        z_n_t
    );

    return std::make_pair(
        camera_pos, 
        camera_viewpoint
    );
}


__host__ void build_floor(
    polygon* polygons, float3* floor_vertices, uchar4 scene_color, int offset = 0
) {

    polygons[offset] = {
        floor_vertices[0], floor_vertices[1], floor_vertices[2], scene_color
    };

    polygons[offset + 1] = {
        floor_vertices[0], floor_vertices[2], floor_vertices[3], scene_color
    };
}


__host__ void build_hexahedron(
    polygon* polygons,
    float3 center, uchar4 color, float radius,
    int offset = 2
) {

    float side_len = 2.f * radius / std::sqrt(3);
    float x = center.x - side_len / 2.f;
    float y = center.y - side_len / 2.f;
    float z = center.z - side_len / 2.f;

    std::vector<float3> verts;
	verts.push_back(make_float3(x, y, z));
	verts.push_back(make_float3(x + side_len, y, z));
	verts.push_back(make_float3(x, y + side_len, z));
	verts.push_back(make_float3(x, y, z + side_len));
	verts.push_back(make_float3(x + side_len, y + side_len, z));
	verts.push_back(make_float3(x, y + side_len, z + side_len));
	verts.push_back(make_float3(x + side_len, y, z + side_len));
	verts.push_back(make_float3(x + side_len, y + side_len, z + side_len));

    polygons[offset++] = get_polygon(0, 2, 4, verts, color);
    polygons[offset++] = get_polygon(0, 1, 4, verts, color);
    polygons[offset++] = get_polygon(0, 5, 2, verts, color);
    polygons[offset++] = get_polygon(0, 5, 3, verts, color);
    polygons[offset++] = get_polygon(1, 4, 7, verts, color);
    polygons[offset++] = get_polygon(1, 6, 7, verts, color);
    polygons[offset++] = get_polygon(3, 6, 7, verts, color);
    polygons[offset++] = get_polygon(3, 5, 7, verts, color);
    polygons[offset++] = get_polygon(0, 3, 6, verts, color);
    polygons[offset++] = get_polygon(0, 1, 6, verts, color);
    polygons[offset++] = get_polygon(7, 4, 5, verts, color);
    polygons[offset++] = get_polygon(4, 5, 2, verts, color);
}


__host__ void build_dodecahedron(
    polygon* polygons, 
    float3 center, uchar4 color, float radius, 
    int offset = 14
) {
    float phi = (1.f + std::sqrt(5)) / 2.f;
    float a = 1.f / std::sqrt(3);
    float b = a / phi;
    float c = a * phi;

    std::vector<float3> verts;
    std::vector<float> ones = {-1., 1.};

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {

            verts.push_back(make_float3(
                center.x, 
                center.y + ones[i] * c * radius, 
                center.z + ones[j] * b * radius
            ));

            verts.push_back(make_float3(
                center.x + ones[i] * c * radius, 
                center.y + ones[j] * b * radius, 
                center.z
            ));

            verts.push_back(make_float3(
                center.x + ones[i] * b * radius, 
                center.y, 
                center.z + ones[j] * c * radius
            ));

            for (int k = 0; k < 2; ++k) {
                verts.push_back(make_float3(
                    center.x + ones[i] * a * radius, 
                    center.y + ones[j] * a * radius, 
                    center.z + ones[k] * a * radius
                ));
            }

        }
    }

    polygons[offset++] = get_polygon(0, 1, 3, verts, color);
    polygons[offset++] = get_polygon(0, 1, 4, verts, color);
    polygons[offset++] = get_polygon(0, 4, 5, verts, color);
    polygons[offset++] = get_polygon(0, 5, 14, verts, color);
    polygons[offset++] = get_polygon(0, 11, 14, verts, color);
    polygons[offset++] = get_polygon(0, 11, 13, verts, color);
    polygons[offset++] = get_polygon(0, 12, 13, verts, color);
    polygons[offset++] = get_polygon(0, 12, 2, verts, color);
    polygons[offset++] = get_polygon(0, 2, 3, verts, color);
    polygons[offset++] = get_polygon(11, 14, 17, verts, color);
    polygons[offset++] = get_polygon(11, 17, 16, verts, color);
    polygons[offset++] = get_polygon(16, 17, 19, verts, color);
    polygons[offset++] = get_polygon(17, 19, 15, verts, color);
    polygons[offset++] = get_polygon(17, 9, 15, verts, color);
    polygons[offset++] = get_polygon(17, 9, 7, verts, color);
    polygons[offset++] = get_polygon(17, 7, 4, verts, color);
    polygons[offset++] = get_polygon(17, 4, 5, verts, color);
    polygons[offset++] = get_polygon(17, 14, 5, verts, color);
    polygons[offset++] = get_polygon(3, 1, 6, verts, color);
    polygons[offset++] = get_polygon(3, 6, 8, verts, color);
    polygons[offset++] = get_polygon(3, 8, 2, verts, color);
    polygons[offset++] = get_polygon(7, 9, 6, verts, color);
    polygons[offset++] = get_polygon(7, 6, 1, verts, color);
    polygons[offset++] = get_polygon(7, 1, 4, verts, color);
    polygons[offset++] = get_polygon(9, 15, 10, verts, color);
    polygons[offset++] = get_polygon(9, 10, 8, verts, color);
    polygons[offset++] = get_polygon(9, 8, 6, verts, color);
    polygons[offset++] = get_polygon(18, 10, 8, verts, color);
    polygons[offset++] = get_polygon(18, 8, 2, verts, color);
    polygons[offset++] = get_polygon(18, 2, 12, verts, color);
    polygons[offset++] = get_polygon(18, 16, 11, verts, color);
    polygons[offset++] = get_polygon(18, 11, 13, verts, color);
    polygons[offset++] = get_polygon(18, 13, 12, verts, color);
    polygons[offset++] = get_polygon(18, 16, 19, verts, color);
    polygons[offset++] = get_polygon(18, 19, 15, verts, color);
    polygons[offset++] = get_polygon(18, 15, 10, verts, color);
}


__host__ void build_icosahedron(
    polygon* polygons, 
    float3 center, uchar4 color, float radius, 
    int offset = 50
) {

    float phi = (1.f + std::sqrt(5)) / 2.f;

    std::vector<float3> verts;
    std::vector<float> ones = {-1., 1.};

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {

            verts.push_back(make_float3(
                center.x, 
                center.y + ones[i] * (1 / 2.f) * radius, 
                center.z + ones[j] * (phi / 2.f) * radius
            ));

            verts.push_back(make_float3(
                center.x + ones[i] * (1 / 2.f) * radius, 
                center.y + ones[j] * (phi / 2.f) * radius, 
                center.z           
            ));

            verts.push_back(make_float3(
                center.x + ones[i] * (phi / 2.f) * radius, 
                center.y, 
                center.z + ones[j] * (1 / 2.f) * radius
            ));
        }
    }

    polygons[offset++] = get_polygon(9, 3, 11, verts, color);
    polygons[offset++] = get_polygon(9, 3, 5, verts, color);
    polygons[offset++] = get_polygon(9, 5, 4, verts, color);
    polygons[offset++] = get_polygon(9, 4, 10, verts, color);
    polygons[offset++] = get_polygon(9, 10, 11, verts, color);
    polygons[offset++] = get_polygon(1, 5, 3, verts, color);
    polygons[offset++] = get_polygon(1, 3, 7, verts, color);
    polygons[offset++] = get_polygon(1, 7, 0, verts, color);
    polygons[offset++] = get_polygon(1, 0, 2, verts, color);
    polygons[offset++] = get_polygon(1, 2, 5, verts, color);
    polygons[offset++] = get_polygon(6, 2, 4, verts, color);
    polygons[offset++] = get_polygon(6, 4, 10, verts, color);
    polygons[offset++] = get_polygon(6, 10, 8, verts, color);
    polygons[offset++] = get_polygon(6, 8, 0, verts, color);
    polygons[offset++] = get_polygon(6, 0, 2, verts, color);
    polygons[offset++] = get_polygon(8, 11, 7, verts, color);
    polygons[offset++] = get_polygon(8, 0, 7, verts, color);
    polygons[offset++] = get_polygon(8, 10, 11, verts, color);
    polygons[offset++] = get_polygon(11, 3, 7, verts, color);
    polygons[offset++] = get_polygon(2, 5, 4, verts, color); 
}


__host__ void build_scene(
    polygon* polygons,
    float3* floor_vertices, uchar4 scene_color,
    float3 hexahedron_center, uchar4 hexahedron_color, float hexahedron_radius,
    float3 dodecahedron_center, uchar4 dodecahedron_color, float dodecahedron_radius,
    float3 icosahedron_center, uchar4 icosahedron_color, float icosahedron_radius
) {

    build_floor(polygons, floor_vertices, scene_color);

    build_hexahedron(
        polygons, hexahedron_center, hexahedron_color, hexahedron_radius
    );

    build_dodecahedron(
        polygons, dodecahedron_center, dodecahedron_color, dodecahedron_radius 
    );

    build_icosahedron(
        polygons, icosahedron_center, icosahedron_color, icosahedron_radius
    );
}


__host__ void ssaa_cpu(
    uchar4* data, uchar4* ssaa_data, 
    int width, int height, int value_for_ssaa
) {

    int r, g, b, alpha = 0;
    int filter_size = value_for_ssaa * value_for_ssaa;

    for (int y = 0; y < height; y += 1) {
        for (int x = 0; x < width; x += 1) {
            r = 0;
            g = 0;
            b = 0;
            
            for (int i = 0; i < value_for_ssaa; ++i) {
                for (int j = 0; j < value_for_ssaa; ++j){
                    uchar4 p = data[
                        value_for_ssaa * value_for_ssaa * width * y + value_for_ssaa * width * j + value_for_ssaa * x + i 
                    ];
                    r += p.x;
                    g += p.y;
                    b += p.z;
                }
            }
            r /= filter_size;
            g /= filter_size;
            b /= filter_size;

			ssaa_data[y * width + x] = make_uchar4(r, g, b, alpha);
        }
    }
}


__global__ void ssaa_gpu(
    uchar4* data, uchar4* ssaa_data, 
    int width, int height, int value_for_ssaa
) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    int r, g, b, alpha = 0;
    int filter_size = value_for_ssaa * value_for_ssaa;

    for (int y = idy; y < height; y += offsety) {
        for (int x = idx; x < width; x += offsetx) {
            r = 0;
            g = 0;
            b = 0;

            for (int i = 0; i < value_for_ssaa; i++) {
				for (int j = 0; j < value_for_ssaa; j++) {
                    uchar4 p = data[
                        value_for_ssaa * value_for_ssaa * width * y + value_for_ssaa * width * j + value_for_ssaa * x + i
                    ];
					r += p.x;
					g += p.y;
					b += p.z;
				}
			}

            r /= filter_size;
            g /= filter_size;
            b /= filter_size;

			ssaa_data[y * width + x] = make_uchar4(r, g, b, alpha);
        }
    }
}


__device__ __host__ uchar4 ray(
    float3 pos, float3 dir, 
    float3 light_src, float3 light_source_color_normalized, 
    polygon* polygons, int num_triangles
) {

    float ts_min;
    int k_min = -1;

    for (int k = 0; k < num_triangles; k++) {
        float3 e1 = diff(polygons[k].y, polygons[k].x);
        float3 e2 = diff(polygons[k].z, polygons[k].x);
        float3 p = prod(dir, e2);
        float div = dot(p, e1);
        if (std::fabs(div) < 1e-10) {
            continue;
        }

        float3 t = diff(pos, polygons[k].x);
        float u = dot(p, t) / div;
        if (u < 0.f || u > 1.f) {
            continue;
        }

        float3 q = prod(t, e1);
        float v = dot(q, dir) / div;
        if (v < 0.f || v + u > 1.f) {
            continue;
        }
        
        float ts = dot(q, e2) / div;
        if (ts < 0.f) {
            continue;
        }

        if (k_min == -1 || ts < ts_min) {
            k_min = k;
            ts_min = ts;
        }
    }

    if (k_min == -1) {
        return make_uchar4(0, 0, 0, 0);
    }

    pos.x = dir.x * ts_min + pos.x;
	pos.y = dir.y * ts_min + pos.y;
	pos.z = dir.z * ts_min + pos.z;

    dir = diff(light_src, pos);
    float len = std::sqrt(dot(dir, dir));
    dir = norm(dir);

    for (int k = 0; k < num_triangles; k++) {
        float3 e1 = diff(polygons[k].y, polygons[k].x);
        float3 e2 = diff(polygons[k].z, polygons[k].x);
        float3 p = prod(dir, e2);
        float div = dot(p, e1);
        if (std::fabs(div) < 1e-10) {
            continue;
        }

        float3 t = diff(pos, polygons[k].x);
        float u = dot(p, t) / div;
        if (u < 0.f || u > 1.f) {
            continue;
        }
        
        float3 q = prod(t, e1);
        float v = dot(q, dir) / div;
        if (v < 0.f || v + u > 1.f) {
            continue;
        }
 
        float ts = dot(q, e2) / div;
        if (ts > 0.f && ts < len && k != k_min) {
            return make_uchar4(0, 0, 0, 0);
        }
    }

    uchar4 res = polygons[k_min].color;
    res.x = res.x * light_source_color_normalized.x;
    res.y = res.y * light_source_color_normalized.y;
    res.z = res.z * light_source_color_normalized.z;
    return res;
}


__host__ void render_cpu(
    int width, int height, int fov, 
    float3 camera_pos, float3 camera_viewpoint, 
    float3 light_src, float3 light_source_color_normalized,
    uchar4* data, 
    polygon* polygons, int num_triangles
) {
    
    float dw = 2.f / (width - 1.f);
    float dh = 2.f / (height - 1.f);
    float z = 1.f / std::tan(fov * M_PI / 360.f);
    float3 bz = norm(diff(camera_viewpoint, camera_pos));
    float3 bx = norm(prod(bz, make_float3(0, 0, 1)));
    float3 by = norm(prod(bx, bz));

    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            float3 v = make_float3(
                -1.f + dw * i,
                (-1.f + dh * j) * height / width,
                z
            );

            float3 dir = mult(bx, by, bz, v);
            data[(height - 1 - j) * width + i] = ray(
                camera_pos, 
                norm(dir), 
                light_src, light_source_color_normalized, 
                polygons, num_triangles
            );
        }
    }
}


__global__ void render_gpu(
    int width, int height, int fov, 
    float3 camera_pos, float3 camera_viewpoint, 
    float3 light_src, float3 light_source_color_normalized,
    uchar4* data, 
    polygon* polygons, int num_triangles
) {
    
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    float dw = 2.f / (width - 1.f);
    float dh = 2.f / (height - 1.f);
    float z = 1.f / std::tan(fov * M_PI / 360.f);
    float3 bz = norm(diff(camera_viewpoint, camera_pos));
    float3 bx = norm(prod(bz, make_float3(0, 0, 1)));
    float3 by = norm(prod(bx, bz));
    for (int j = idy; j < height; j += offsety) {
        for (int i = idx; i < width; i += offsetx) {

            float3 v = make_float3(
                -1.f + dw * i, 
                (-1.f + dh * j) * height / width, 
                z
            );

            float3 dir = mult(bx, by, bz, v);
            data[(height - 1 - j) * width + i] = ray(
                camera_pos, norm(dir), 
                light_src, light_source_color_normalized, 
                polygons, num_triangles
            );
        }
    }
}


__host__ void run(
    std::string device,
    std::pair<float3, float3> camera_settings,
    int width, int height, int value_for_ssaa, int fov,
    float3 light_source_pos, float3 light_source_color_normalized,
    int num_triangles, polygon* polygons, polygon* polygons_dev,
    uchar4* data, uchar4* ssaa_data, uchar4* data_dev, uchar4* ssaa_data_dev
) {

    float3 camera_pos = camera_settings.first;
    float3 camera_viewpoint = camera_settings.second;

    if (device == "cpu") {
        render_cpu(
            width * value_for_ssaa, height * value_for_ssaa, fov, 
            camera_pos, camera_viewpoint, 
            light_source_pos, light_source_color_normalized, 
            data, 
            polygons, num_triangles
        );
        ssaa_cpu(
            data, ssaa_data, 
            width, height, 
            value_for_ssaa
        );
        memcpy(data, ssaa_data, width * height * sizeof(uchar4));
        return;
    }

    render_gpu<<<dim3(16, 32), dim3(16, 32)>>>(
        width * value_for_ssaa, height * value_for_ssaa, fov, 
        camera_pos, camera_viewpoint, 
        light_source_pos, light_source_color_normalized, 
        data_dev, 
        polygons_dev, num_triangles
    );
    CSC(cudaGetLastError());

    ssaa_gpu<<<dim3(16, 32), dim3(16, 32)>>>(
        data_dev, ssaa_data_dev, 
        width, height, 
        value_for_ssaa
    );
    CSC(cudaGetLastError());

    CSC(cudaMemcpy(
        data, 
        ssaa_data_dev, 
        width * height * sizeof(uchar4),
        cudaMemcpyDeviceToHost
    ));
}


__host__ void print_settings(
    int n_frames,
    std::string imagepaths,
    int height, int width, int fov,
    float r_0_c, float z_0_c, float phi_0_c,
    float A_r_c, float A_z_c,
    float omega_r_c, float omega_z_c, float omega_phi_c,
    float p_r_c, float p_z_c,
    float r_0_n, float z_0_n, float phi_0_n,
    float A_r_n, float A_z_n,
    float omega_r_n, float omega_z_n, float omega_phi_n,
    float p_r_n, float p_z_n,
    float3 hexahedron_center, float3 hexahedron_color_normalized, float hexahedron_radius,
    float3 dodecahedron_center, float3 dodecahedron_color_normalized, float dodecahedron_radius,
    float3 icosahedron_center, float3 icosahedron_color_normalized, float icosahedron_radius,
    float3* floor_points, float3 scene_color_normalized,
    float3 light_source_pos, float3 light_source_color_normalized,
    int value_for_ssaa
) {

    using namespace std;

    cout << "Settings:" << endl;
    cout << n_frames << endl;
    cout << imagepaths << endl;
    cout << height << ' ' <<  width << ' ' << fov << endl;
    cout << r_0_c << ' ' << z_0_c << ' ' << phi_0_c << '\t' << A_r_c << ' ' << A_z_c << '\t' << omega_r_c << ' ' << omega_z_c << ' ' << omega_phi_c << '\t' << p_r_c << ' ' << p_z_c << endl;
    cout << r_0_n << ' ' << z_0_n << ' ' << phi_0_n << '\t' << A_r_n << ' ' << A_z_n << '\t' << omega_r_n << ' ' << omega_z_n << ' ' << omega_phi_n << '\t' << p_r_n << ' ' << p_z_n << endl;
    cout << hexahedron_center.x << ' ' << hexahedron_center.y << ' ' << hexahedron_center.z << '\t' << hexahedron_color_normalized.x << ' ' << hexahedron_color_normalized.y << ' ' << hexahedron_color_normalized.z << '\t' << hexahedron_radius << endl;
    cout << dodecahedron_center.x << ' ' << dodecahedron_center.y << ' ' << dodecahedron_center.z << '\t' << dodecahedron_color_normalized.x << ' ' << dodecahedron_color_normalized.y << ' ' << dodecahedron_color_normalized.z << '\t' << dodecahedron_radius << endl;
    cout << icosahedron_center.x << ' ' << icosahedron_center.y << ' ' << icosahedron_center.z << '\t' << icosahedron_color_normalized.x << ' ' << icosahedron_color_normalized.y << ' ' << icosahedron_color_normalized.z << '\t' << icosahedron_radius << endl;
    cout << floor_points[0].x << ' ' << floor_points[0].y << ' ' << floor_points[0].z << '\t' << floor_points[1].x << ' ' << floor_points[1].y << ' ' << floor_points[1].z << '\t' << floor_points[2].x << ' ' << floor_points[2].y << ' ' << floor_points[2].z << '\t' << floor_points[3].x << ' ' << floor_points[3].y << ' ' << floor_points[3].z << " no_data " << scene_color_normalized.x << ' ' << scene_color_normalized.y << ' ' << scene_color_normalized.z << endl;
    cout << 1 << endl;
    cout << light_source_pos.x << ' ' << light_source_pos.y << ' ' << light_source_pos.z << '\t' << light_source_color_normalized.x << ' ' << light_source_color_normalized.y << ' ' << light_source_color_normalized.z << endl;
    cout << 0 << ' ' << value_for_ssaa << endl;
}


int main(int argc, char *argv[]) {
    std::string device = "gpu";
    bool set_default_settings = true;

    if (argc > 1) {
        std::string input = std::string(argv[1]);
        if (input == "--gpu") {
            device = "gpu";
            set_default_settings = false;
        } else if (input == "--cpu") {
            device = "cpu";
            set_default_settings = false;
        } else if (input == "--default") {
            device = "gpu";
            set_default_settings = true;
        } else {
            std::cout << "Suggested keys: --gpu, --cpu, --default";
            return 0;
        }
    }

    int n_frames = 1;
    std::string imagepaths = "./output/%d.data";
    int width = 1920, height = 1080, fov = 80;

    float r_0_c = 6.5, z_0_c = 6., phi_0_c = 0.3;
    float A_r_c = 0.5, A_z_c = 0.;
    float omega_r_c = 0.5, omega_z_c = 1., omega_phi_c = 0.7;
    float p_r_c = 2., p_z_c = 5.;

    float r_0_n = 1.5, z_0_n = 0., phi_0_n = 0.;
    float A_r_n = 0.3, A_z_n = 0.05;
    float omega_r_n = 1., omega_z_n = 1., omega_phi_n = 0.7;
    float p_r_n = 1., p_z_n = 1.;

    float3 hexahedron_center = make_float3(-2.5, -1.5, 1.5);
    float3 hexahedron_color_normalized = make_float3(1., 0., 0.);
    float hexahedron_radius = 1;

    float3 dodecahedron_center = make_float3(0., 0., 1.5);
    float3 dodecahedron_color_normalized = make_float3(0., 0.7, 0.);
    float dodecahedron_radius = 1;

    float3 icosahedron_center = make_float3(2.5, 1.5, 1.5);
    float3 icosahedron_color_normalized = make_float3(0., 0., 1.);
    float icosahedron_radius = 1;

    float3 floor_vertices[4];
    floor_vertices[0] = make_float3(-15., -15., 0.);
    floor_vertices[1] = make_float3(-15., 15., 0.);
    floor_vertices[2] = make_float3(15., 15., 0.);
    floor_vertices[3] = make_float3(15., -15., 0.);

    float3 scene_color_normalized = make_float3(0.3, 0.7, 0.7);
    float3 light_source_pos = make_float3(10., 30., 25.);
    float3 light_source_color_normalized = make_float3(1., 1., 1.);

    int value_for_ssaa = 3;

    float not_implemented_float;
    std::string not_implemented_str;

    if (set_default_settings == false) {
        std::cin >> n_frames;
        std::cin >> imagepaths;
        std::cin >> width >> height >> fov;

        std::cin >> r_0_c >> z_0_c >> phi_0_c >> A_r_c >> A_z_c >> omega_r_c >> omega_z_c >> omega_phi_c >> p_r_c >> p_z_c;
        std::cin >> r_0_n >> z_0_n >> phi_0_n >> A_r_n >> A_z_n >> omega_r_n >> omega_z_n >> omega_phi_n >> p_r_n >> p_z_n;

        std::cin >> hexahedron_center.x >> hexahedron_center.y >> hexahedron_center.z;
        std::cin >> hexahedron_color_normalized.x >> hexahedron_color_normalized.y >> hexahedron_color_normalized.z;
        std::cin >> hexahedron_radius;
        std::cin >> not_implemented_float >> not_implemented_float >> not_implemented_float; 

        std::cin >> dodecahedron_center.x >> dodecahedron_center.y >> dodecahedron_center.z;
        std::cin >> dodecahedron_color_normalized.x >> dodecahedron_color_normalized.y >> dodecahedron_color_normalized.z;
        std::cin >> dodecahedron_radius;
        std::cin >> not_implemented_float >> not_implemented_float >> not_implemented_float;

        std::cin >> icosahedron_center.x >> icosahedron_center.y >> icosahedron_center.z;
        std::cin >> icosahedron_color_normalized.x >> icosahedron_color_normalized.y >> icosahedron_color_normalized.z;
        std::cin >> icosahedron_radius;
        std::cin >> not_implemented_float >> not_implemented_float >> not_implemented_float;

        for (int i = 0; i < 4; ++i) {
            std::cin >> floor_vertices[i].x >> floor_vertices[i].y >> floor_vertices[i].z;
        }
        std::cin >> not_implemented_str;
        std::cin >> scene_color_normalized.x >> scene_color_normalized.y >> scene_color_normalized.z;
        std::cin >> not_implemented_float;

        // 1 light source
        std::cin >> not_implemented_float;
        std::cin >> light_source_pos.x >> light_source_pos.y >> light_source_pos.z;
        std::cin >> light_source_color_normalized.x >> light_source_color_normalized.y >> light_source_color_normalized.z;

        std::cin >> not_implemented_float;
        std::cin >> value_for_ssaa;
    } else {
        print_settings(
            n_frames,
            imagepaths,
            height, width, fov,
            r_0_c, z_0_c, phi_0_c,
            A_r_c, A_z_c,
            omega_r_c, omega_z_c, omega_phi_c,
            p_r_c, p_z_c,
            r_0_n, z_0_n, phi_0_n,
            A_r_n, A_z_n,
            omega_r_n, omega_z_n, omega_phi_n,
            p_r_n, p_z_n,
            hexahedron_center, hexahedron_color_normalized, hexahedron_radius,
            dodecahedron_center, dodecahedron_color_normalized, dodecahedron_radius,
            icosahedron_center, icosahedron_color_normalized, icosahedron_radius,
            floor_vertices, scene_color_normalized,
            light_source_pos, light_source_color_normalized,
            value_for_ssaa
        );
    }

    uchar4 hexahedron_color = color_to_uchar4(hexahedron_color_normalized);
    uchar4 dodecahedron_color = color_to_uchar4(dodecahedron_color_normalized);
    uchar4 icosahedron_color = color_to_uchar4(icosahedron_color_normalized);
    uchar4 scene_color = color_to_uchar4(scene_color_normalized);

    int num_triangles = 70;
    polygon polygons[num_triangles];

    uchar4* data = new uchar4[width * height * value_for_ssaa * value_for_ssaa];
    uchar4* ssaa_data = new uchar4[width * height];

    polygon* polygons_dev;
    uchar4 *data_dev, *ssaa_data_dev;
    if (device == "gpu") {
        CSC(cudaMalloc(&polygons_dev, num_triangles * sizeof(polygon)));
        CSC(cudaMalloc(
            &data_dev, 
            value_for_ssaa * value_for_ssaa * width * height * sizeof(uchar4)
        ));
        CSC(cudaMalloc(&ssaa_data_dev, width * height * sizeof(uchar4)));
    }

    build_scene(
        polygons,
        floor_vertices,
        scene_color,
        hexahedron_center, hexahedron_color, hexahedron_radius,
        dodecahedron_center, dodecahedron_color, dodecahedron_radius,
        icosahedron_center, icosahedron_color, icosahedron_radius
    );

    if (device == "gpu") {
        CSC(cudaMemcpy(
            polygons_dev, 
            polygons, 
            num_triangles * sizeof(polygon), 
            cudaMemcpyHostToDevice
        ));
    }

    for (int frame = 0; frame < n_frames; ++frame) {

        auto start_time = std::chrono::high_resolution_clock::now();

        float current_step = 2.f * M_PI * frame / n_frames;

        std::pair<float3, float3> camera_settings = get_camera_settings(
            current_step,
            r_0_c, z_0_c, phi_0_c,
            A_r_c, A_z_c,
            omega_r_c, omega_z_c, omega_phi_c,
            p_r_c, p_z_c,
            r_0_n, z_0_n, phi_0_n,
            A_r_n, A_z_n,
            omega_r_n, omega_z_n, omega_phi_n,
            p_r_n, p_z_n
        );

        run(
            device,
            camera_settings,
            width, height, value_for_ssaa, fov,
            light_source_pos, light_source_color_normalized,
            num_triangles, polygons, polygons_dev,
            data, ssaa_data, data_dev, ssaa_data_dev
        );

        char output_imagepath[300];
        std::sprintf(output_imagepath, imagepaths.c_str(), frame);

        FILE *out = fopen(output_imagepath, "w");
        std::fwrite(&width, sizeof(width), 1, out);
        std::fwrite(&height, sizeof(height), 1, out);
        std::fwrite(
            data, 
            sizeof(uchar4), 
            width * height, 
            out
        );
        std::fclose(out);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto time = end_time - start_time;
        
        int total_rays = height * width * value_for_ssaa * value_for_ssaa;
        std::cout << frame << '\t' << time/std::chrono::milliseconds(1) << '\t' << total_rays << std::endl;
    }

    if (device == "gpu") {
        CSC(cudaFree(data_dev));
        CSC(cudaFree(ssaa_data_dev));
        CSC(cudaFree(polygons_dev));
    }

    free(data);
    free(ssaa_data);
}