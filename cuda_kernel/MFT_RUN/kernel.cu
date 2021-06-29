#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <direct.h>
#include <cfloat>
#include <limits>
#include <stdexcept>
#include <ctime>
#include <chrono>
// max_element
#include <thrust/extrema.h>

#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>


// for minus
#include <thrust/sequence.h>
#include <thrust/fill.h>
#include <thrust/transform.h>



#include <thrust/inner_product.h>
#include <cmath>
//#include <C:/Program Files/boost/boost_1_62_0/boost/filesystem/path.hpp>
//////////////////////////////////////////

// CUDA Utils

#pragma once

#ifndef CUDA_UTIL_CUH_
#define CUDA_UTIL_CUH_

#include <cinttypes>
#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define CUDA_VERIFY(value) {\
	cudaError_t _m_cudaStat = value;\
	if (_m_cudaStat != cudaSuccess) {\
		fprintf(stderr, "Error %s at line %d in file %s\n", cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);\
		exit(1);\
	}\
}

#define CUDA_VERIFY_KERNEL(value) {\
	value;\
	cudaError_t _m_cudaStat = cudaGetLastError();\
	if (_m_cudaStat != cudaSuccess) {\
		fprintf(stderr, "Error %s at line %d in file %s\n", cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);\
		exit(1);\
	}\
}

struct CudaParameters
{
	uint32_t device_number; // graphic card number
	uint32_t grid_size; // belongs to gr. card description - block number
	uint32_t block_size; // belongs to gr. card description - thread number
};

struct Shape
{
	uint32_t dim;
	uint32_t size;
	uint32_t x, y, z;

	Shape() {}
	Shape(uint32_t x_) : x(x_) { dim = 1; size = x; y = 1; z = 1; }
	Shape(uint32_t x_, uint32_t y_) : x(x_), y(y_) { dim = 2; size = x * y; z = 1; }
	Shape(uint32_t x_, uint32_t y_, uint32_t z_) : x(x_), y(y_), z(z_) { dim = 3; size = x * y * z; }
	~Shape() {}
};

template<typename T>
struct DeviceArray
{
	Shape shape;
	T *p;
};

template <typename T>
DeviceArray<T> set_dev_arr(T value, Shape shape)
{
	DeviceArray<T> darr;
	darr.shape = shape;
	CUDA_VERIFY(cudaMalloc((void**)&darr.p, shape.size * sizeof(T)));
	CUDA_VERIFY(cudaMemset(darr.p, value, shape.size * sizeof(T)));
	return darr;
}

template <typename T>
DeviceArray<T> vec_to_dev_arr(std::vector<T> &arr, Shape shape)
{
	DeviceArray<T> darr;
	darr.shape = shape;
	CUDA_VERIFY(cudaMalloc((void**)&darr.p, arr.size() * sizeof(T))); // allocate memory in graphic card
	CUDA_VERIFY(cudaMemcpy(darr.p, &arr[0], arr.size() * sizeof(T), cudaMemcpyHostToDevice)); // copy data from 'arr' into 'darr'
	return darr;
}

template<typename T>
std::vector<T> dev_arr_to_vec(DeviceArray<T> &darr)
{
	std::vector<T> arr(darr.shape.size);
	CUDA_VERIFY(cudaMemcpy(&arr[0], darr.p, darr.shape.size * sizeof(T), cudaMemcpyDeviceToHost));
	return arr;
}

template<typename T>
T* set_dev_ptr(T value)
{
	T* ptr;
	CUDA_VERIFY(cudaMalloc((void**)&ptr, sizeof(T)));
	CUDA_VERIFY(cudaMemset(ptr, value, sizeof(T)));
	return ptr;
}

template<typename T>
T dev_ptr_to_var(T *dvar)
{
	T var = (T)0;
	CUDA_VERIFY(cudaMemcpy(&var, dvar, sizeof(T), cudaMemcpyDeviceToHost));
	return var;
}

#endif /* CUDA_UTIL_CUH_ */


// Cuda header

#pragma once

#ifndef MFT_CUH_
#define MFT_CUH_

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>

// #include "cuda_util.cuh"
// #include "coord.h" // To delete

namespace voxel_t
{
	enum voxel_types : uint8_t {
		VOID = 0, WALL = 1, BULK = 2, NUM = 3
	};
}

struct Coord
{
	int32_t x, y, z;
};

struct LambdaValue
{
	std::string state;
	float lambda_val;
};

struct SimParams
{
	float Tstar;
	float e_wf;
	float e_ff;
	float precision;
	uint32_t max_iter;
	uint32_t eval_iter;
	uint32_t gpu_device;
	uint32_t gpu_blocks;
	uint32_t gpu_threads;
};

class Mft
{
private:
	uint32_t label;

	CudaParameters cup;

	Coord size;
	std::vector<uint8_t> map;
	DeviceArray<uint8_t> d_map;
	uint32_t num_void;	

	std::vector<float> epsilon;
	DeviceArray<float> d_epsilon;

	float t_star;

	std::vector<float> density;
	DeviceArray<float> d_density;
	DeviceArray<float> d_eval_density;

public:
	std::vector<float> diff_abs_max_vect;

	uint32_t max_iter;
	uint32_t eval_iter;
	float precision;

	float avg_density = 0.0f;
	uint32_t num_iter = 0;

	float tmp = 0.0f;

public:
	Mft(CudaParameters cup_, std::vector<uint8_t> map_, Coord size_, std::vector<float> epsilon_, float t_star);
	~Mft() {}

	float run(float lambda);

	std::vector<float> get_system_state(bool walls = false);
};

#endif /* MFT_CUH_ */

//////////////////////////////////////////
//
// MFT cu

// #include "mft.cuh"

Mft::Mft(CudaParameters cup_,
	std::vector<uint8_t> map_,
	Coord size_,
	std::vector<float> epsilon_,
	float t_star_) :
	cup(cup_), map(map_), size(size_), epsilon(epsilon_), t_star(t_star_)
{
	// label = time(NULL); // delete

	// Init cuda device
	int32_t n_devices = 0;
	CUDA_VERIFY(cudaGetDeviceCount(&n_devices));
	if (cup.device_number < n_devices) {
		CUDA_VERIFY(cudaSetDevice(cup.device_number));
		CUDA_VERIFY(cudaDeviceSynchronize());
	}
	else {
		throw;
	}

	// Copy vectors to device
	d_map = vec_to_dev_arr(map, Shape(size.x, size.y, size.z));
	d_epsilon = vec_to_dev_arr(epsilon, Shape(epsilon.size()));

	density.resize(map.size(), 0.0); // set up vector size and its default values
	d_density = vec_to_dev_arr(density, Shape(density.size()));
	d_eval_density = vec_to_dev_arr(density, Shape(density.size()));

	// Init setup vars
	max_iter = 20000;
	eval_iter = 30;
	precision = 1e-6;

	// # of VOID voxel
	num_void = 0;
	for (auto x : map) {
		if (x == voxel_t::VOID) { num_void++; }
	}
}

std::vector<float> Mft::get_system_state(bool walls)
{
	std::vector<float> sys = density;
	if (walls) {
		for (auto i = 0; i < map.size(); i++) {
			if (map[i] == voxel_t::WALL) {
				sys[i] = -1.0f;
			}
		}
	}
	return sys;
}

/*
CUDA device functions
*/
__device__ int3 index_1d_to_3d(int32_t k, int3 size) {
	return{ k % size.x, (k / size.x) % size.y, k / (size.x * size.y) };
}

__device__ int32_t index_3d_to_1d(int3 c, int3 size) {
	return c.x + c.y * size.x + c.z * size.x * size.y;
}

__device__ void neighbors(int32_t(&n)[6], int32_t k, int3 size) {
	int3 c = index_1d_to_3d(k, size);
	n[0] = index_3d_to_1d({ (((c.x - 1) % size.x) + size.x) % size.x, c.y, c.z }, size);
	n[1] = index_3d_to_1d({ (((c.x + 1) % size.x) + size.x) % size.x, c.y, c.z }, size);
	n[2] = index_3d_to_1d({ c.x, (((c.y - 1) % size.y) + size.y) % size.y, c.z }, size);
	n[3] = index_3d_to_1d({ c.x, (((c.y + 1) % size.y) + size.y) % size.y, c.z }, size);
	n[4] = index_3d_to_1d({ c.x, c.y, (((c.z - 1) % size.z) + size.z) % size.z }, size);
	n[5] = index_3d_to_1d({ c.x, c.y, (((c.z + 1) % size.z) + size.z) % size.z }, size);
}





__global__ void dev_mft(DeviceArray<uint8_t> map, 
	DeviceArray<float> density, 
	DeviceArray<float> epsilon, 
	float t_star,
	float mu)
{
	int3 size = { int32_t(map.shape.x), int32_t(map.shape.y), int32_t(map.shape.z) };

	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	for (auto i = id; i < map.shape.size; i += blockDim.x * gridDim.x) {
		if (map.p[i] != voxel_t::WALL) {

			int32_t n[6];
			neighbors(n, i, size);

			float energy = 0.0f;
			for (auto j = 0; j < 6; j++) {
				if (map.p[n[j]] == voxel_t::WALL) {
					//energy += epsilon.p[voxel_t::WALL];

					/*
					* To exclude wall interactions for the bulk phase
					* outside of your material.  so that the bulk always
					* stays nicely isotropic and is not affected too much by the walls
					*/

					if (map.p[i] != voxel_t::BULK) {
						//energy += epsilon.p[voxel_t::WALL];
						energy += epsilon.p[voxel_t::WALL]/epsilon.p[voxel_t::VOID];
						
					}
				}
				else {
					// energy += epsilon.p[voxel_t::VOID] * density.p[n[j]];
					energy += density.p[n[j]];
				}
			}

			// density.p[i] = 1. / (1. + exp(-energy - mu) );
			density.p[i] = 1. / (1. + exp((-energy - mu) / t_star));
		}
	}
}

__global__ void dev_get_eval_density(DeviceArray<uint8_t> map, DeviceArray<float> density, DeviceArray<float> eval_density)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	for (auto i = id; i < map.shape.size; i += blockDim.x * gridDim.x) {
		if (map.p[i] == voxel_t::VOID) {
			eval_density.p[i] = density.p[i];
		}
	}
}

template<class ArgumentType, class ResultType>
struct unary_function
{
	typedef ArgumentType argument_type;
	typedef ResultType result_type;
};



template<typename T>
struct thrust_abs_val : public unary_function<T, T>
{
	__host__ __device__ T operator()(const T &x) const
	{
		return x < T(0) ? -x : x;
	}
};

template<typename T>
struct thrust_square_val : public unary_function<T, T>
{
	__host__ __device__ T operator()(const T &x) const
	{
		return x*x;
	}
};

template <typename T>
struct thrust_abs_diff : public thrust::binary_function<T, T, T>
{
	__host__ __device__
		T operator()(const T& a, const T& b)
	{
		return fabsf(b - a);
	}
};


/*
CUDA host functions
*/
float Mft::run(float lambda)
{
	assert(lambda > 0.0f);
		
	// Gpu timer init
	float gpu_time;
	cudaEvent_t gpu_start;
	cudaEvent_t gpu_stop;
	cudaEventCreate(&gpu_start); cudaEventCreate(&gpu_stop);
	cudaEventRecord(gpu_start, 0);

	// Set up parameters for MFT model
	// float mu = std::log(lambda) - 3.0f * epsilon[voxel_t::VOID];	
	float mu = t_star * std::log(lambda) - 3.0f;
	uint32_t n_iter = max_iter;
	
	std::vector<float>().swap(diff_abs_max_vect);

	thrust::device_vector<float> thrust_vec_diff(d_density.shape.size);
	thrust::device_vector<float> thrust_vec_dens_old(d_density.shape.size);
	thrust::fill(thrust_vec_diff.begin(), thrust_vec_diff.end(), 0.0f);
	thrust::fill(thrust_vec_dens_old.begin(), thrust_vec_dens_old.end(), 0.0f);
	
	CUDA_VERIFY(cudaDeviceSynchronize());
	for (auto i_iter = 0; i_iter < max_iter; i_iter++) {
		//std::cout << i_iter << std::endl;
		dev_mft <<<cup.grid_size, cup.block_size >>>(d_map, d_density, d_epsilon, t_star, mu);

		if (i_iter % eval_iter == (eval_iter-1) ) {
			thrust::device_ptr<float> thrust_ptr_density(d_density.p);
			thrust::device_vector<float> thrust_vec_density(thrust_ptr_density, thrust_ptr_density + d_density.shape.size);
			
			// max_abs_diff
			float max_abs_diff = thrust::inner_product(
				thrust_vec_density.begin(), 
				thrust_vec_density.end(), 
				thrust_vec_dens_old.begin(),
				0.0f, 
				thrust::maximum<float>(), 
				thrust_abs_diff<float>() );

			diff_abs_max_vect.push_back(max_abs_diff);

			if (max_abs_diff < precision) {
				n_iter = i_iter+1;
				break;
			}
			thrust::copy(thrust::device, thrust_vec_density.begin(), thrust_vec_density.end(), thrust_vec_dens_old.begin());
		}
	}

	// Eval
	// CUDA_VERIFY_KERNEL((dev_get_eval_density<<<cup.grid_size, cup.block_size>>>(d_map, d_density, d_eval_density)));
	dev_get_eval_density << <cup.grid_size, cup.block_size >> > (d_map, d_density, d_eval_density);
	thrust::device_ptr<float> thrust_ptr(d_eval_density.p);
	thrust::device_vector<float> thrust_vec(thrust_ptr, thrust_ptr + d_eval_density.shape.size);
	avg_density = thrust::reduce(thrust_vec.begin(), thrust_vec.end(), 0.0f, thrust::plus<float>()) / num_void;
	CUDA_VERIFY(cudaDeviceSynchronize());
	num_iter = n_iter;

	// Copy density to host
	density = dev_arr_to_vec(d_density);

	// Gpu timer final
	cudaEventRecord(gpu_stop, 0);
	cudaEventSynchronize(gpu_stop);
	cudaEventElapsedTime(&gpu_time, gpu_start, gpu_stop);
	//std::cout << "  Device time =   " << gpu_time/1000.0f << "\n";
	cudaEventDestroy(gpu_start); cudaEventDestroy(gpu_stop);
	return avg_density;
}

template<typename T>
void read_from_binary(std::vector<T> &vec, std::string path)
{
	FILE *file = fopen(path.c_str(), "r");
	assert(file != NULL);
	fread(vec.data(), sizeof(T), vec.size(), file);
	fclose(file);
}

template<typename T>
void dump_vector_to_binary_file(std::vector<T> &v, std::string path)
{
	std::string temp = path; // +".mftout";
	FILE *f = fopen(temp.c_str(), "wb");
	fwrite(v.data(), sizeof(T), v.size(), f);
	fclose(f);
	std::rename(temp.c_str(), path.c_str());
}

template<typename T>
void read_lambda_vec_file(std::vector<T> &labmda_vec, std::string path )
{
	std::string::size_type sz;
	std::string line;
	std::string state;
	std::string lambda_str_val;
	std::ifstream labmda_file(path);	
	assert(labmda_file.is_open());	
	while (std::getline(labmda_file, line))
	{
		std::stringstream stream_line(line);
		std::getline(stream_line, state, '\t');
		std::getline(stream_line, lambda_str_val, '\t');
		LambdaValue Cur_lambda_val = { state, std::stof(lambda_str_val, &sz) };
		labmda_vec.push_back(Cur_lambda_val);
	}
	labmda_file.close();
}

template<typename T>
void read_map_size_file(std::vector<T> &map_size_vect, std::string path)
{
	std::string line;
	uint32_t size_value;
	std::ifstream myfile(path); 
	if (myfile.is_open())
	{
		getline(myfile, line, '\n');
		while (getline(myfile, line, '\t'))
		{
			size_value = std::stoul(line, nullptr, 0);
			map_size_vect.push_back(size_value);
		}
		myfile.close();
	}
}

template<typename T>
void read_sim_params_file(std::vector<T> &sim_params_vect, std::string path)
{
	std::string line;
	std::string::size_type sz;
	float param_value;
	std::ifstream myfile(path);
	if (myfile.is_open())
	{
		getline(myfile, line, '\n');
		while (getline(myfile, line, '\t'))
		{
			param_value = std::stof(line, &sz);
			sim_params_vect.push_back(param_value);
		}
		myfile.close();
	}
}

template<typename T>
void fill_lambda_vec(std::vector<T> &lambda_vec, T start_val, T stop_val, T step)
{
	std::cout << "_______ADS_______" << std::endl;
	for (float cur_lmda = start_val; cur_lmda <= stop_val + std::numeric_limits<T>::epsilon() * 10; cur_lmda += step)
	{
		std::cout << cur_lmda << std::endl;
		//sprintf(lmda_val_str, "%04d", (int) roundf((cur_lmda * 1000) ));
	}
	std::cout << "_______DES_______" << std::endl;
	for (float cur_lmda = stop_val - step; cur_lmda >= start_val - std::numeric_limits<T>::epsilon() * 10; cur_lmda -= step) {
		std::cout << cur_lmda << std::endl;
	}
}

void write_text_to_log_file(const std::string dest_fpath, const std::string &text)
{
	std::ofstream log_file(dest_fpath, 
		std::ios_base::out | std::ios_base::app);
	log_file << text << std::endl;
}

template<typename T>
void write_vector_to_log_file(const std::string dest_fpath, const std::vector<T> vec)
{
	std::ofstream log_file(dest_fpath,
		std::ios_base::out | std::ios_base::app);

	

	for (auto i = 0; i < vec.size() - 1; i++)
	{
		log_file << vec[i] << '\t';
	}
	log_file << vec[vec.size()-1] << std::endl;
}
//////////////////////////////////////////

int main(int argc, char ** argv)
{
	int32_t arg_x = 0;
	int32_t arg_y = 0;
	int32_t arg_z = 0;

	double total_time = 0.0;

	std::string simulation_dirpath = "";
	
	if (argc == 1)
	{
		simulation_dirpath = "d:/1_Work/PROJECTS/2019/07_Physisorption/IMAGES_TO_SIM/SBA15_1div3/simulation_00000"; 		
	}
	else if(argc == 2)
	{
		simulation_dirpath = argv[1];
	}
	else
	{
		std::cerr << "Error: incorrect amount of CMD arguments" << std::endl;
		return EXIT_FAILURE;
	}
	
	
	std::string log_string = "";
	std::string diff_abs_max_string = "";

	std::string log_written_files = simulation_dirpath + "/log_written_files.txt";
	std::string log_compl_lams = simulation_dirpath + "/log_completed_lambdas.txt";
	std::string log_sim_progress = simulation_dirpath + "/log_sim_progress.txt";
	std::string map_fpath = simulation_dirpath + "/../map.mftin";
	std::string simresbin_dirpath = simulation_dirpath + "/sim_outfiles_bin";
	std::string lambda_fpath = simulation_dirpath + "/lambda_values.mftin";
	std::string map_size_fpath = simulation_dirpath + "/../map_size.mftin";
	std::string sim_params_fpath = simulation_dirpath + "/sim_params.mftin";

	std::vector<LambdaValue> lambda_vect;
	std::vector<uint32_t> map_size_vect;
	std::vector<float> sim_params_vect;
	std::vector<float> epsilon = { 0.0f, 0.0f }; // eff, efw
	float t_star = 0.0; // reduced temperature
	Coord size_str = { 0, 0, 0 };
	SimParams Sim_params = { 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0 };

	read_lambda_vec_file(lambda_vect, lambda_fpath);
	read_sim_params_file(sim_params_vect, sim_params_fpath);
	read_map_size_file(map_size_vect, map_size_fpath);


	
	_mkdir(simresbin_dirpath.c_str());

	// python writes an array in a different order
	// therefore indices are in reverse order
	arg_x = map_size_vect[2];
	arg_y = map_size_vect[1];
	arg_z = map_size_vect[0];
	
	Sim_params.Tstar = sim_params_vect[0];
	Sim_params.e_wf = sim_params_vect[1];
	Sim_params.e_ff = sim_params_vect[2];
	Sim_params.precision = sim_params_vect[3];
	Sim_params.max_iter = sim_params_vect[4];
	Sim_params.eval_iter = sim_params_vect[5];
	Sim_params.gpu_device = sim_params_vect[6];
	Sim_params.gpu_blocks = sim_params_vect[7];
	Sim_params.gpu_threads = sim_params_vect[8];

	size_str = { arg_x, arg_y, arg_z };	
	epsilon = { Sim_params.e_ff, Sim_params.e_wf }; // eff, efw
	t_star = Sim_params.Tstar; // reduced temperature
		
	std::vector<uint8_t> objvect(size_str.x * size_str.y * size_str.z);
	read_from_binary(objvect, map_fpath);
	
	CudaParameters cudpar = { Sim_params.gpu_device,
		Sim_params.gpu_blocks,
		Sim_params.gpu_threads }; // gpu device number
											  // , number of blocks of gpu, threads per block 
	
	
	Mft Mft_obj(cudpar, objvect, size_str, epsilon, t_star);
	
	Mft_obj.precision = Sim_params.precision;
	Mft_obj.max_iter = Sim_params.max_iter;
	Mft_obj.eval_iter = Sim_params.eval_iter;
	
	time_t current_time = time(0);
	log_string = "Simulation folder:\n" +
		simulation_dirpath +
		'\n' +
		ctime(&current_time) +
		'\n' +
		"_______________________________________________\n" +
		"Header:\nstate\tlambda_value\ttotal_time_hrs\ttotal_time_min\ttotal_time_seconds\n" +
		"_______RUNNING..._______";
	write_text_to_log_file(log_sim_progress, log_string);

	

	for (auto cur_lmda : lambda_vect) 
	{	
		auto time_0 = std::chrono::steady_clock::now();

		/*
		std::cout
			<< Mft_obj.num_iter
			<< '\t'
			<< Mft_obj.run(cur_lmda.lambda_val)
			<< std::endl;
		*/

		Mft_obj.run(cur_lmda.lambda_val);

		auto time_1 = std::chrono::steady_clock::now();
		double elapsed_seconds = std::chrono::duration_cast<
			std::chrono::duration<double> >(time_1 - time_0).count();
		total_time += elapsed_seconds;

		log_string = "-------" +
			cur_lmda.state +
			'\t' +
			std::to_string(cur_lmda.lambda_val) +
			'\t' +
			std::to_string(elapsed_seconds/3600) +
			'\t' +
			std::to_string(elapsed_seconds /60) +
			'\t' +
			std::to_string(elapsed_seconds) +
			"-------";

		write_text_to_log_file(log_sim_progress, log_string);
		write_vector_to_log_file(log_sim_progress,
			Mft_obj.diff_abs_max_vect);

		log_string = cur_lmda.state +
			'\t' +
			std::to_string(cur_lmda.lambda_val)			
			;
		write_text_to_log_file(log_compl_lams, log_string);


		std::string str_lambda_val = 
			std::to_string((int16_t)(cur_lmda.lambda_val * 1000));
		std::string simresbin_fpath = simresbin_dirpath +
			"/activity_lambda_" +
			cur_lmda.state +
			"_" +
			std::string(4-str_lambda_val.length(), '0') +
			str_lambda_val +
			".mftout";
		//std::cout << simresbin_fpath << std::endl;
		
		std::vector<float> curr_state = Mft_obj.get_system_state(2);
		dump_vector_to_binary_file(curr_state, simresbin_fpath);
		write_text_to_log_file(log_written_files, simresbin_fpath);
	}
	
	log_string = "\n-------TOTAL_TIME" +
		'\t' +		
		std::to_string(total_time / 3600) +
		'\t' +
		std::to_string(total_time / 60) +
		'\t' +
		std::to_string(total_time) +
		"-------";
	write_text_to_log_file(log_sim_progress, log_string);
	
	std::cout << '\n';
	return 0;
}
