// ** Trial for my Personal Reference **


#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cassert>

// Similiar to Matrix Multiplication
// There we use a 2D grid and 2D blocks
// O(bcd), so can be parallelized
__global__ void horner(const int *d_coeffs,const int *d_domain,const int *d_results,int d_coeffs_size,int d_batch_size,int d_domain_size){
    // 
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Global index for the batch
    for (uint64_t j = 0; j < d_domain_size; ++j) {
        if (i < d_batch_size) {
            // Each thread computes the polynomial for one batch
            int result = d_coeffs[i * d_coeffs_size + d_coeffs_size - 1]; // Start with the last coefficient
            for (int k = d_coeffs_size - 2; k >= 0; --k) {
                result = result * d_domain[i * d_domain_size + j] + d_coeffs[i * d_coeffs_size + k];
            }
            d_results[i * d_domain_size + j] = result; // Store the result
        }
    }
}



int main(){
    // These are our constraints 
    int coeffs_size = 1 << 10;
    int batch_size = 10;
    int domain_size = 7;
    

    int coeffs[coeffs_size * batch_size];
    for (int i = 0; i < coeffs_size * batch_size; ++i) {
        coeffs[i] = i % 100; // Example coefficients
    }
    int domain[domain_size * batch_size];
    for (int i = 0; i < domain_size * batch_size; ++i) {
        domain[i] = i % 100; // Example domain values
    }

    int results[batch_size * domain_size];

    size_t coeff_size = coeffs_size * sizeof(int);
    size_t domain_size = domain_size * sizeof(int);
    size_t batch_size = batch_size * sizeof(int);
    size_t domain_size_bytes = domain_size * batch_size * sizeof(int);
    size_t coeffs_size_bytes = coeffs_size * batch_size * sizeof(int);
    
    int *d_coeffs, *d_domain, *d_results, *d_coeffs_size, *d_domain_size, *d_batch_size;
    
    cudaMalloc(&d_coeffs, coeffs_size_bytes);
    cudaMalloc(&d_domain, domain_size_bytes);
    cudaMalloc(&d_results, domain_size_bytes);
    cudaMalloc(&d_coeffs_size, sizeof(int));
    cudaMalloc(&d_domain_size, sizeof(int));
    cudaMalloc(&d_batch_size, sizeof(int));

    cudaMemcpy(d_coeffs, coeffs, coeffs_size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_domain, domain, domain_size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_coeffs_size, &coeffs_size, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_domain_size, &domain_size, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_batch_size, &batch_size, sizeof(int), cudaMemcpyHostToDevice); 

    // Launch the kernel 

    int threads_per_block = 256; // Number of threads per block
    int num_blocks = (batch_size + threads_per_block - 1) / threads_per
    horner<<<num_blocks, threads_per_block>>>(d_coeffs, d_domain, d_results, coeffs_size, batch_size, domain_size);
    cudaDeviceSynchronize(); // Wait for the kernel to finish
    cudaMemcpy(results, d_results, domain_size_bytes, cudaMemcpyDeviceToHost);
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < domain_size; ++j) {
            std::cout << "Result[" << i << "][" << j << "] = " << results[i * domain_size + j] << std::endl;
        }
    }
}