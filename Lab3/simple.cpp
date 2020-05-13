//
// Created by Андрей Гайдамака on 24.04.2020.
//
#include <iostream>

#define M 4
#define N 4
#define K 4


void print_matrix(double *matrix, int rows, int cols){
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix[i*cols + j];
        }
        std::cout << std::endl;
    }
}

void matrixMultiply(const double * a, const double * b, double *result) {
  for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
          for (int k = 0; k < K; ++k) {
              result[i * K + k] += a[i * N + j] * b[j * K + k];
          }
      }
  }

}

void fill(double *matrix, int rows, int cols) {
  for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
          matrix[i * cols + j] = i * cols + j;
      }
  }
}

int main(int argc, char *argv[]) {

    double *A = new double[M * N];
    double *B = new double[K * N];
    double *C = new double[K * M];
    fill(A, M, N);
    fill(B, N, K);
    matrixMultiply(A, B, C);
    print_matrix(C, M, K);
    delete [] A;
    delete [] B;
    delete [] C;
    return 0;
}