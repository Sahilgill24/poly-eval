#include <cuda_runtime.h>

template <typename T>
__global__ void horner_kernel(
    const T *__restrict__ coeffs,
    const T *__restrict__ domain,
    int coeffs_size,
    int domain_size,
    int batch_size,
    T *__restrict__ evals /*OUT*/)
{

  int idx_in_domain = blockIdx.x * blockDim.x + threadIdx.x;
  int poly = blockIdx.y;
  if (idx_in_domain >= domain_size)
    return;

  const T *curr_coeffs = coeffs + poly * coeffs_size;
  T *curr_evals = evals + poly * domain_size;

  curr_evals[idx_in_domain] = curr_coeffs[coeffs_size - 1];
  for (int coeff_idx = coeffs_size - 2; coeff_idx >= 0; --coeff_idx)
  {
    curr_evals[idx_in_domain] =
        curr_evals[idx_in_domain] * domain[idx_in_domain] + curr_coeffs[coeff_idx];
  }
}

template <typename T>
__global__ void horner_kernel_v1(
    const T *__restrict__ coeffs,
    const T *__restrict__ domain,
    int coeffs_size,
    int domain_size,
    int batch_size,
    T *__restrict__ evals /*OUT*/)
{
  int idx_in_batch = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx_in_batch >= batch_size)
    return;

  const T *curr_coeffs = coeffs + idx_in_batch * coeffs_size;
  T *curr_evals = evals + idx_in_batch * domain_size;

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

template <typename T>
void poly_eval_ref(
    const T *coeffs,
    const T *domain,
    int coeffs_size,
    int domain_size,
    int batch_size,
    T *evals /*OUT*/)
{
  // using Horner's method
  // example: ax^2+bx+c is computed as (1) r=a, (2) r=r*x+b, (3) r=r*x+c
  for (uint64_t idx_in_batch = 0; idx_in_batch < batch_size; ++idx_in_batch)
  {
    const T *curr_coeffs = coeffs + idx_in_batch * coeffs_size;
    T *curr_evals = evals + idx_in_batch * domain_size;
    for (uint64_t eval_idx = 0; eval_idx < domain_size; ++eval_idx)
    {
      curr_evals[eval_idx] = curr_coeffs[coeffs_size - 1];
      for (int64_t coeff_idx = coeffs_size - 2; coeff_idx >= 0; --coeff_idx)
      {
        curr_evals[eval_idx] =
            curr_evals[eval_idx] * domain[eval_idx] + curr_coeffs[coeff_idx];
      }
    }
  }
}

template <typename T>
void poly_eval(
    const T *coeffs,
    const T *domain,
    int coeffs_size,
    int domain_size,
    int batch_size,
    T *evals /*OUT*/)
{
  dim3 grid_size((domain_size + 255) / 256, batch_size);
  dim3 block_size(256);
  horner_kernel<T><<<grid_size, block_size>>>(coeffs, domain, coeffs_size, domain_size, batch_size, evals);
}