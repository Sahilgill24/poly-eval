// ** Basic implementation of polynomial evaluation for my personal reference **
// Honner
#include <stdio.h>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>


class Scalar
{
public:
    static const unsigned int MODULUS = 1000000007;

    unsigned x;

    Scalar(unsigned long long x) : x(x % MODULUS) {}
    Scalar() : x(0) {}
    // basic operator Overloading for the Scalar Class
    // does all operations in the field defined by MODULUS
    Scalar operator+(const Scalar &other) const
    {
        return Scalar((x + other.x) % MODULUS);
    }

    Scalar operator-(const Scalar &other) const
    {
        return Scalar((x - other.x + MODULUS) % MODULUS);
    }

    Scalar operator*(const Scalar &other) const
    {
        return Scalar(((unsigned long long)x * other.x) % MODULUS);
    }

    Scalar pow(unsigned exp) const
    {
        Scalar base = *this;
        Scalar result(1);
        while (exp > 0)
        {
            if (exp % 2 == 1)
            {
                result = result * base;
            }
            base = base * base;
            exp /= 2;
        }
        return result;
    }
    Scalar swap2(Scalar &other)
    {
        std::swap(x, other.x);
        return *this;
    }

};
// very similiar to Matrix multiplication, but we are just using loops for multiplication here.
// need to think parallely for this ....
// already given in the Challenge statement.
void poly_eval_ref_honnor(
    const Scalar *coeffs,
    const Scalar *domain,
    int coeffs_size,
    int domain_size,
    int batch_size,
    Scalar *evals /*OUT*/)
{
    // using Horner's method
    // example: ax^2+bx+c is computed as (1) r=a, (2) r=r*x+b, (3) r=r*x+c
    for (uint64_t idx_in_batch = 0; idx_in_batch < batch_size; ++idx_in_batch)
    {
        const Scalar *curr_coeffs = coeffs + idx_in_batch * coeffs_size;
        Scalar *curr_evals = evals + idx_in_batch * domain_size;
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

int main()
{
    // Example test
    auto start = std::chrono::high_resolution_clock::now();
    const int coeffs_size = 2;
    const int domain_size = 2;
    const int batch_size = 2;

    Scalar coeffs[batch_size * coeffs_size] = {
        Scalar(1), Scalar(2), Scalar(3), Scalar(4)};
    Scalar domain[domain_size] = {
        Scalar(4), Scalar(5)};
    Scalar evals[batch_size * domain_size] = {
        Scalar(0), Scalar(0), Scalar(0), Scalar(0) // Placeholder for results
    };

    poly_eval_ref_honnor(coeffs, domain, coeffs_size, domain_size, batch_size, evals);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    printf("Time taken: %f seconds\n", elapsed.count() * 100000);
    for (int i = 0; i < batch_size * domain_size; ++i)
    {
        printf("evals[%d] = %u\n", i, evals[i].x);
    }

    return 0;
}
