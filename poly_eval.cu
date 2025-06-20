template <typename T>
void poly_eval_ref(
  const T* coeffs,
  const T* domain,
  int coeffs_size,
  int domain_size,
  int batch_size,
  T* evals /*OUT*/)
{
  // using Horner's method
  // example: ax^2+bx+c is computed as (1) r=a, (2) r=r*x+b, (3) r=r*x+c
  for (uint64_t idx_in_batch = 0; idx_in_batch < batch_size; ++idx_in_batch) {
    const T* curr_coeffs = coeffs + idx_in_batch * coeffs_size;
    T* curr_evals = evals + idx_in_batch * domain_size;
    for (uint64_t eval_idx = 0; eval_idx < domain_size; ++eval_idx) {
      curr_evals[eval_idx] = curr_coeffs[coeffs_size - 1];
      for (int64_t coeff_idx = coeffs_size - 2; coeff_idx >= 0; --coeff_idx) {
        curr_evals[eval_idx] =
          curr_evals[eval_idx] * domain[eval_idx] + curr_coeffs[coeff_idx];
      }
    }
  }
}


template <typename T>
void poly_eval(
  const T* coeffs,
  const T* domain,
  int coeffs_size,
  int domain_size,
  int batch_size,
  T* evals /*OUT*/)
{
  /*implement your solution here*/
}