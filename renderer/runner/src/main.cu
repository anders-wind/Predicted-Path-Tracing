#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <limits>
#include <random>
#include "include/vec3.cuh"
#include "include/ray.cuh"
#include "include/sphere.cuh"
#include "include/hitable.cuh"
#include "include/hitable_list.cuh"
#include "include/camera.cuh"
#include "include/material.cuh"
#include "include/random_helpers.cuh"
#include "include/cuda_renderer.cuh"

#define RM(row, col, w) row *w + col
#define CM(row, col, h) col *h + row
//
//void write_ppm_image(std::vector<rgb> colors, int h, int w, std::string filename) {
//	std::ofstream myfile;
//	myfile.open(filename + ".ppm");
//	myfile << "P3\n" << w << " " << h << "\n255\n";
//	for (int i = 0; i < h; i++) {
//		for (int j = 0; j < w; j++) {
//			auto color = colors[RM(i, j, w)];
//			myfile << color.r()*255.99 << " " << color.g()*255.99 << " " << color.b()*255.99 << std::endl;
//		}
//	}
//	myfile.close();
//}
//
//std::vector<rgb> hello_world_render(int h, int w) {
//	auto colors = std::vector<rgb>(w*h);
//	for (int i = 0; i < h; i++) {
//		for (int j = 0; j < w; j++) {
//			colors[RM(i, j, w)].r(j / float(w));
//			colors[RM(i, j, w)].g(h - i / float(h));
//			colors[RM(i, j, w)].b(0.2f);
//		}
//	}
//	return colors;
//}
//
//rgb color(const ray& r, const std::shared_ptr<hitable>& world, int depth) {
//	hit_record rec;
//	if (world->hit(r, 0.001f, std::numeric_limits<float>::max(), rec)) {
//		ray scattered;
//		vec3 attenuation;
//		if (depth < 50 && rec.mat_ptr->scatter(r, rec, attenuation, scattered)) {
//			return attenuation * color(scattered, world, depth + 1);
//		}
//		else {
//			return rgb(0, 0, 0);
//		}
//	}
//	vec3 unit_direction = unit_vector(r.direction());
//	float t = 0.5f*(unit_direction.e[1] + 1.0f);
//	return vec3(1.0f, 1.0f, 1.0f)*(1.0f - t) + vec3(0.5f, 0.7f, 1.0f)*t;
//}
//
//std::vector<rgb> simple_ray_render(int h, int w, int samples) {
//	auto colors = std::vector<rgb>(w*h);
//	auto c = camera();
//	auto world = std::make_shared<hitable_list>();
//	world->add_hitable(std::make_shared<sphere>(vec3(0, 0, -1), 0.5f, std::make_shared<lambertian>(vec3(0.8f, 0.3f, 0.3f))));
//	world->add_hitable(std::make_shared<sphere>(vec3(0, -100.5, -1), 100.0f, std::make_shared<lambertian>(vec3(0.8f, 0.8f, 0.0f))));
//	world->add_hitable(std::make_shared<sphere>(vec3(1, 0, -1), 0.5f, std::make_shared<metal>(vec3(0.8f, 0.6f, 0.2f), 0.3f)));
//	world->add_hitable(std::make_shared<sphere>(vec3(-1, 0, -1), 0.5f, std::make_shared<dielectric>(1.5f)));
//	world->add_hitable(std::make_shared<sphere>(vec3(-1, 0, -1), -0.45f, std::make_shared<dielectric>(1.5f)));
//
//	for (int i = 0; i < h; i++) {
//		for (int j = 0; j < w; j++) {
//			rgb pix(0, 0, 0);
//			for (int s = 0; s < samples; s++) {
//				float u = float(j + dis(gen)) / float(w);
//				float v = float(h - i + dis(gen)) / float(h);
//				ray r = c.get_ray(u, v);
//				pix += color(r, world, 0);
//			}
//			pix /= float(samples);
//			pix = pix.v_sqrt(); // gamma correct (gamma 2)
//			colors[RM(i, j, w)] = pix;
//		}
//	}
//	return colors;
//}
//
//std::vector<rgb> cuda_ray_render(int h, int w, int samples) {
//	size_t fb_size = 3 * h*w * sizeof(float);
//	float *fb;
//	checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));
//
//	auto colors = std::vector<rgb>(w*h);
//	auto c = camera();
//	auto world = std::make_shared<hitable_list>();
//	world->add_hitable(std::make_shared<sphere>(vec3(0, 0, -1), 0.5f, std::make_shared<lambertian>(vec3(0.8f, 0.3f, 0.3f))));
//	world->add_hitable(std::make_shared<sphere>(vec3(0, -100.5, -1), 100.0f, std::make_shared<lambertian>(vec3(0.8f, 0.8f, 0.0f))));
//	world->add_hitable(std::make_shared<sphere>(vec3(1, 0, -1), 0.5f, std::make_shared<metal>(vec3(0.8f, 0.6f, 0.2f), 0.3f)));
//	world->add_hitable(std::make_shared<sphere>(vec3(-1, 0, -1), 0.5f, std::make_shared<dielectric>(1.5f)));
//	world->add_hitable(std::make_shared<sphere>(vec3(-1, 0, -1), -0.45f, std::make_shared<dielectric>(1.5f)));
//
//	for (int i = 0; i < h; i++) {
//		for (int j = 0; j < w; j++) {
//			rgb pix(0, 0, 0);
//			for (int s = 0; s < samples; s++) {
//				float u = float(j + dis(gen)) / float(w);
//				float v = float(h - i + dis(gen)) / float(h);
//				ray r = c.get_ray(u, v);
//				pix += color(r, world, 0);
//			}
//			pix /= float(samples);
//			pix = pix.v_sqrt(); // gamma correct (gamma 2)
//			colors[RM(i, j, w)] = pix;
//		}
//	}
//	return colors;
//}

int main()
{
	int w = 1200;
	int h = 600;
	int s = 400;

	try
	{
		auto colors = cuda_renderer::cuda_ray_render(w, h, s);
		cuda_renderer::write_ppm_image(colors, w, h, "render");
	}
	catch (...)
	{
		std::cout << "failed" << std::endl;
	}
}