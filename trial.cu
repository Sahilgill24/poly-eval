// ** Trial for my Personal Reference **

#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cassert>
#include <cstdlib>
#include <cstdio>

// Similiar to Matrix Multiplication
// There we use a 2D grid and 2D blocks
// O(bcd), so can be parallelized
#define HOST_DEVICE_INLINE __host__ __device__ __forceinline__
#define HOST_INLINE __host__ __forceinline__

#define P_MOD 32843
template <unsigned P = P_MOD>

class Dummy_Scalar
{
public:
    static constexpr unsigned MODULUS = P;

    unsigned x;

    static HOST_DEVICE_INLINE Dummy_Scalar zero() { return {0}; }

    static HOST_DEVICE_INLINE Dummy_Scalar one() { return {1}; }

    friend HOST_INLINE std::ostream &operator<<(std::ostream &os, const Dummy_Scalar &scalar)
    {
        os << scalar.x;
        return os;
    }

    friend HOST_DEVICE_INLINE Dummy_Scalar operator+(Dummy_Scalar p1, const Dummy_Scalar &p2)
    {
        return {(p1.x + p2.x) % MODULUS};
    }

    friend HOST_DEVICE_INLINE Dummy_Scalar operator*(Dummy_Scalar p1, const Dummy_Scalar &p2)
    {
        return {(p1.x * p2.x) % MODULUS};
    }

    friend HOST_DEVICE_INLINE bool operator==(const Dummy_Scalar &p1, const Dummy_Scalar &p2)
    {
        return (p1.x == p2.x);
    }

    friend HOST_DEVICE_INLINE bool operator!=(const Dummy_Scalar &p1, const Dummy_Scalar &p2)
    {
        return (p1.x != p2.x);
    }

    friend HOST_DEVICE_INLINE bool operator==(const Dummy_Scalar &p1, const unsigned p2)
    {
        return (p1.x == p2);
    }

    static HOST_DEVICE_INLINE Dummy_Scalar neg(const Dummy_Scalar &scalar)
    {
        return {MODULUS - scalar.x};
    }

    static HOST_INLINE Dummy_Scalar rand_host()
    {
        return {static_cast<unsigned>(rand() % MODULUS)};
    }
};

typedef Dummy_Scalar<> test_scalar;

__global__ void horner_kernel(const test_scalar *coeffs,
                              const test_scalar *domain,
                              int coeffs_size,
                              int domain_size,
                              int batch_size,
                              test_scalar *evals)
{
    int idx_in_batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx_in_batch >= batch_size)
        return;

    const test_scalar *curr_coeffs = coeffs + idx_in_batch * coeffs_size;
    test_scalar *curr_evals = evals + idx_in_batch * domain_size;

    for (int eval_idx = 0; eval_idx < domain_size; ++eval_idx)
    {
        curr_evals[eval_idx] = curr_coeffs[coeffs_size - 1];
        for (int coeff_idx = coeffs_size - 2; coeff_idx >= 0; --coeff_idx)
        {
            curr_evals[eval_idx] =
                curr_evals[eval_idx] * domain[eval_idx] + curr_coeffs[coeff_idx];
        }
    }
}
// one includes the kernel without the polynomials being batched.
// for (uint64_t idx_in_batch = 0; idx_in_batch < batch_size; ++idx_in_batch)
// {
//     const Scalar *curr_coeffs = coeffs + idx_in_batch * coeffs_size;
//     Scalar *curr_evals = evals + idx_in_batch * domain_size;
//     for (uint64_t eval_idx = 0; eval_idx < domain_size; ++eval_idx)
//     {
//         curr_evals[eval_idx] = curr_coeffs[coeffs_size - 1];
//         for (int64_t coeff_idx = coeffs_size - 2; coeff_idx >= 0; --coeff_idx)
//         {
//             curr_evals[eval_idx] =
//                 curr_evals[eval_idx] * domain[eval_idx] + curr_coeffs[coeff_idx];
//         }
//     }
// }
__global__ void horner_kernel2() {}

void poly_eval(
    const test_scalar *coeffs,
    const test_scalar *domain,
    int coeffs_size,
    int domain_size,
    int batch_size,
    test_scalar *evals /*OUT*/)
{
    dim3 grid_size((batch_size + 255) / 256);
    dim3 block_size(256);
    horner_kernel<<<grid_size, block_size>>>(coeffs, domain, coeffs_size, domain_size, batch_size, evals);
}

void polyeval2(
    const test_scalar *coeffs,
    const test_scalar *domain,
    int coeffs_size,
    int domain_size,
    int batch_size,
    test_scalar *evals /*OUT*/)
{
    // using the kernel here
}

int main()
{
    cudaEvent_t start, stop;
    float time;

    // test parameters
    int coeffs_log_size = 22;
    int coeffs_size = 1 << coeffs_log_size;
    int domain_size = 7;
    int batch_size = 10;
    int total_coeffs_size = batch_size * coeffs_size;
    int total_results_size = batch_size * domain_size;

    printf("running poly eval of degree, 2^%d, domain_size=%d, batch_size=%d, scalar modulus=%d\n", coeffs_log_size, domain_size, batch_size, P_MOD);

    // init inputs
    test_scalar *coeffs = new test_scalar[total_coeffs_size];
    test_scalar *domain = new test_scalar[domain_size];
    for (int i = 0; i < total_coeffs_size; i++)
    {
        coeffs[i] = test_scalar::rand_host();
    }
    for (int i = 0; i < domain_size; i++)
    {
        domain[i] = test_scalar::rand_host();
    }

    std::cout << "finished generating inputs" << std::endl;

    test_scalar *results = new test_scalar[total_results_size];
    test_scalar *ref_results = new test_scalar[total_results_size];

    // allocate memory and copy to device
    test_scalar *d_coeffs;
    test_scalar *d_domain;
    test_scalar *d_results;
    cudaMalloc(&d_coeffs, sizeof(test_scalar) * total_coeffs_size);
    cudaMalloc(&d_domain, sizeof(test_scalar) * domain_size);
    cudaMalloc(&d_results, sizeof(test_scalar) * total_results_size);
    cudaMemcpy(d_coeffs, coeffs, sizeof(test_scalar) * total_coeffs_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_domain, domain, sizeof(test_scalar) * domain_size, cudaMemcpyHostToDevice);

    std::cout << "finished copying to device" << std::endl;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    // warm up
    poly_eval(d_coeffs, d_domain, coeffs_size, domain_size, batch_size, d_results);
    cudaDeviceSynchronize();

    cudaEventRecord(start, 0);
    poly_eval(d_coeffs, d_domain, coeffs_size, domain_size, batch_size, d_results);
    cudaEventRecord(stop, 0);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&time, start, stop);
    printf("runtime : %.3f ms.\n", time);

    // run reference and check correctness

    cudaMemcpy(results, d_results, sizeof(test_scalar) * total_results_size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < total_results_size; i++)
    {
        std::cout << "Results" << results[i] << std::endl;
    }

    return 0;
}
