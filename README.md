# Polynomial Evaluation Challenge

In many zero-knowledge (ZK) protocols, we need to evaluate polynomials defined over a scalar finite field. Your task is to implement an efficient solution in CUDA that can evaluate multiple polynomials over the same domain.

For example, evaluate the polynomials P₀(x), P₁(x), ..., Pₙ₋₁(x) over the domain {4, 7, 9}. Here, `n` is the `batch_size`, `domain_size` is 3, and `coeffs_size` represents the shared degree of the polynomials.

You need to implement the function `poly_eval()` in the file `poly_eval.cu`. To test and bench - run `make test_poly_eval`. The function should run for any valid input parameters but the typical parameters are those given in the test file.

Good luck!


## My Solution 

> Run the Makefile to test the solution. 

This is the link to my <a href="https://hackmd.io/@Ayhm2-FHQhSLkoc1OEcr9g/HyaScCbVxg">Solution</a>