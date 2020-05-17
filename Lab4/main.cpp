#include <iostream>
#include<cmath>
#include <mpi.h>

#define E 1e-8
#define A 1e5

int prev;
int next;
int f;
double D[3] = {2, 2, 2};
int N[3] = {300, 300, 300};
double h[3];
double h2[3];
int X;
int Y;
int Z;
double f_i, f_j, f_k;


double phi(double x, double y, double z) {
  return x * x + y * y + z * z;
}

double ro(double x, double y, double z) {
  return 6 - phi(x, y, z) * A;
}

void findDifference(double **F, const int *shifts, int rank, const int *linesPerProc) {
    double max = 0;
    double tmp = 0;
    for (int i = 1; i < linesPerProc[rank] - 2; i++) {
        for (int j = 1; j < N[1]; j++) {
            for (int k = 1; k < N[2]; k++) {
                if ((tmp = fabs(F[next][i * Y * Z + j * Z + k] - phi((i + shifts[rank]) * h[0], j * h[1], k * h[2]))) > max) {
                    max = tmp;
                }
            }
        }
    }
    double resMax = 0;
    MPI_Allreduce(&max, &resMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout<<resMax<<std::endl;
    }
}

void condition(double ** F, int i, int j, int k){
    if (fabs(F[next][i * Y * Z + j * Z + k] - F[prev][i * Y * Z + j * Z + k]) > E) {
        f = 0;
    }
}

void initBounds(double **F, const int *offsets, int rank, const int *linesPerProc) {
    int startLine = offsets[rank];
  for (int i = 0; i < linesPerProc[rank]; i++, startLine++) {
    for (int j = 0; j < N[1]; j++) {
      for (int k = 0; k < N[2]; k++) {
        if ((startLine != 0) && (j != 0) && (k != 0) && (startLine != N[0]-1) && (j != N[1]-1) && (k != N[2]-1)) {
          F[0][i * Y * Z + j * Z + k] = 0;
          F[1][i * Y * Z + j * Z + k] = 0;
        } else {
          F[0][i * Y * Z + j * Z + k] = phi(startLine * h[0], j * h[1], k * h[2]);
          F[1][i * Y * Z + j * Z + k] = F[0][i * Y * Z + j * Z + k];
        }
      }
    }
  }
}

void insideCalculate(double **F, const int *shifts, int rank, double coefficient, const int *linesPerProc) {
    for (int i = 1; i < linesPerProc[rank] - 1; ++i) {
        for (int j = 1; j < N[1]-1; ++j) {
            for (int k = 1; k < N[2]-1; ++k) {
                f_i = (F[prev][(i + 1) * Y * Z + j * Z + k] + F[prev][(i - 1) * Y * Z + j * Z + k]) / h2[0];
                f_j = (F[prev][i * Y * Z + (j + 1) * Z + k] + F[prev][i * Y * Z + (j - 1) * Z + k]) / h2[1];
                f_k = (F[prev][i * Y * Z + j * Z + (k + 1)] + F[prev][i * Y * Z + j * Z + (k - 1)]) / h2[2];
                F[next][i * Y * Z + j * Z + k] = coefficient * (f_i + f_j + f_k - ro((i + shifts[rank]) * h[0], j * h[1], k * h[2]));
                condition(F, i, j, k);
            }
        }
    }
}

void borderCalculate(double **F, double **buffer, double coefficient, int rank, int numOfProc, const int *linesPerProc,
                     const int *shifts) {
  for (int j = 1; j < N[1]-1; ++j) {
    for (int k = 1; k < N[2]-1; ++k) {
      if (rank > 0) {
        int i = 0;
          f_i = (F[prev][(i + 1) * Y * Z + j * Z + k] + buffer[0][j * Z + k]) / h2[0];
          f_j = (F[prev][i * Y * Z + (j + 1) * Z + k] + F[prev][i * Y * Z + (j - 1) * Z + k]) / h2[1];
          f_k = (F[prev][i * Y * Z + j * Z + (k + 1)] + F[prev][i * Y * Z + j * Z + (k - 1)]) / h2[2];
        F[next][i * Y * Z + j * Z + k] = coefficient * (f_i + f_j + f_k - ro((i + shifts[rank]) * h[0], j * h[1], k * h[2]));
        condition(F, i, j, k);
      }
      if (rank < numOfProc - 1) {
        int i = linesPerProc[rank] - 1;
          f_i = (buffer[1][j * Z + k] + F[prev][(i - 1) * Y * Z + j * Z + k]) / h2[0];
          f_j = (F[prev][i * Y * Z + (j + 1) * Z + k] + F[prev][i * Y * Z + (j - 1) * Z + k]) / h2[1];
          f_k = (F[prev][i * Y * Z + j * Z + (k + 1)] + F[prev][i * Y * Z + j * Z + (k - 1)]) / h2[2];
        F[next][i * Y * Z + j * Z + k] = coefficient * (f_i + f_j + f_k - ro((i + shifts[rank]) * h[0], j * h[1], k * h[2]));
        condition(F, i, j, k);
      }
    }
  }
}

void asyncSend(double **F, double **buffer, int rank, const int *linesPerProc, int numOfProc, MPI_Request *sendRequest,
               MPI_Request *receiveRequest) {
  if (rank > 0) { //если не нулевой
    MPI_Isend(&(F[prev][0]), Z * Y, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &sendRequest[0]);
    MPI_Irecv(buffer[0], Z * Y, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &receiveRequest[1]);
  }
  if (rank < numOfProc - 1) { //если не последний
    MPI_Isend(&(F[prev][(linesPerProc[rank] - 1) * Y * Z]),Z * Y, MPI_DOUBLE,rank + 1,1, MPI_COMM_WORLD, &sendRequest[1]);
    MPI_Irecv(buffer[1], Z * Y, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &receiveRequest[0]);
  }
}

void receive(int rank, int numOfProc, MPI_Request *sendRequest, MPI_Request *receiveRequest) {
    if (rank > 0) { // если не нулевой
        MPI_Wait(&receiveRequest[1], MPI_STATUS_IGNORE);
        MPI_Wait(&sendRequest[0], MPI_STATUS_IGNORE);
    }
    if (rank < numOfProc - 1) { //если не последний
        MPI_Wait(&receiveRequest[0], MPI_STATUS_IGNORE);
        MPI_Wait(&sendRequest[1], MPI_STATUS_IGNORE);
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int numOfProc = 0;
    int rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &numOfProc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int procNumMoreOnLine = numOfProc - (N[0] % numOfProc);
    int shiftForIProc = 0;
    int * linesPerProc = new int[numOfProc]();
    int * shifts = new int[numOfProc]();
    int tmp = N[0] / numOfProc;
    for (int i = 0; i < numOfProc; ++i) {
        shifts[i] = shiftForIProc;
        linesPerProc[i] = tmp;
        if (i > procNumMoreOnLine) {
            linesPerProc[i]++;
        }
        shiftForIProc += linesPerProc[i];
    }

    X = linesPerProc[rank];
    Y = N[1] ;
    Z = N[2];
    double * F[2];
    F[0] = new double[X * Y * Z]();
    F[1] = new double[X * Y * Z]();
    double * buffer[2];
    buffer[0] = new double[Z * Y]();
    buffer[1] = new double[Z * Y]();
    for (int i = 0; i < 3; ++i){
        h[i] = D[i] / (N[i] - 1);
        h2[i] = h[i] * h[i];
    }
    double coefficient = 1 / (2 / h2[0] + 2 / h2[1] + 2 / h2[2] + A);
    initBounds(F, shifts, rank, linesPerProc);
    double start = MPI_Wtime();
    MPI_Request sendRequest[2] = {};
    MPI_Request receiveRequest[2] = {};
    f = 0;
    int minF;
    prev = 1;
    next = 0;
    while (!f){
        f = 1;
        prev = 1 - prev; // или 0 или 1
        next = 1 - next; // или 1 или 0
        asyncSend(F, buffer, rank, linesPerProc, numOfProc, sendRequest, receiveRequest);
        insideCalculate(F, shifts, rank, coefficient, linesPerProc);
        receive(rank, numOfProc, sendRequest, receiveRequest);
        borderCalculate(F, buffer, coefficient, rank, numOfProc, linesPerProc, shifts);
        MPI_Allreduce(&f, &minF, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        f = minF;
    }
    if (rank == 0) {
        double finish = MPI_Wtime();
        std::cout << "Time: "<< finish - start << std::endl;
    }
    findDifference(F, shifts, rank, linesPerProc);
    delete[] buffer[0];
    delete[] buffer[1];
    delete[] F[0];
    delete[] F[1];
    delete[] shifts;
    delete[] linesPerProc;
    MPI_Finalize();
    return 0;
}


