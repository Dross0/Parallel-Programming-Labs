#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <mpi.h>
#include <iostream>

#define SIZE 35000
#define E 0.00001
#define MAX_ITERATIONS 10000


double * getArray(int size){
    double * arr = (double *) calloc(size, sizeof(double));
    return arr;
}

double scalar(double * v1, double *v2, int size) {
    double result = 0;
    for (int i = 0; i < size; i++) {
        result += v1[i] * v2[i];
    }
    return result;
}

double norm(double *vector) {
    double result = 0;
    for (int i = 0; i < SIZE; i++) {
        result += vector[i] * vector[i];
    }
    return sqrt(result);
}

void fillTestDataMPI(double *matrixMpi, double *b, double *x, int shift, int numberOfElem) {

    for (int i = 0; i < numberOfElem; i++) {
        for (int j = 0; j < SIZE; j++) {
            matrixMpi[i * SIZE + j] = 1.0;
            if (shift + i == j)
                ++matrixMpi[i * SIZE + j];
        }
        b[i] = SIZE + 1;
        x[i] = rand() % 15;
    }

}


void calculateMatrixOnVector(double *A_MPI, double *v, double *res, int shift, int numberOfElem) {
    for (int i = 0; i < numberOfElem; i++){
        res[i] = scalar(&A_MPI[i * SIZE], v, SIZE);
    }
}

void printVector(double * v, int size){
    for (int i = 0; i < size; ++i){
        printf("%f ", v[i]);
    }
    printf("\n");
}

int main(int argc, char *argv[]) {
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);//получение числа процессов
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);//получение номера процесса


    int *numberOfElements = (int *) calloc(size, sizeof(int));//количество элементов в процессе

    int *shift = (int *) calloc(size, sizeof(int));//сдвиг

    for (int i = 0; i < size; ++i) {
        numberOfElements[i] = (SIZE / size) + ((i < SIZE % size) ? (1) : (0));
    }
    for (int i = 1; i < size; ++i) {
        shift[i] = shift[i - 1] + numberOfElements[i - 1];
    }

    double b[SIZE] = {0};
    double x[SIZE] = {0};
    double * B_MPI = getArray(numberOfElements[rank]);
    double *X_MPI = getArray(numberOfElements[rank]);
    double *A_MPI = getArray(SIZE * numberOfElements[rank]);
    double *y_MPI = getArray(numberOfElements[rank]);
    double * AX_MPI = getArray(numberOfElements[rank]);
    double *AY_MPI = getArray(numberOfElements[rank]);
    double *y = getArray(SIZE);
    fillTestDataMPI(A_MPI, B_MPI, X_MPI, shift[rank], numberOfElements[rank]);
    MPI_Allgatherv(X_MPI, numberOfElements[rank], MPI_DOUBLE, x, numberOfElements, shift, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgatherv(B_MPI, numberOfElements[rank], MPI_DOUBLE, b, numberOfElements, shift, MPI_DOUBLE, MPI_COMM_WORLD);
    double normB = norm(b);
    double tetta = 0;
    double numerator = 0;
    double denominator = 0;
    double denominatorSum = 0;
    double numeratorSum = 0;
    double condition = 100;
    int iterations = 0;

    double startTime = MPI_Wtime();
    while (condition > E && iterations++ < MAX_ITERATIONS) {
        numerator = 0;
        numeratorSum = 0;
        denominator = 0;
        denominatorSum = 0;
        calculateMatrixOnVector(A_MPI, x, AX_MPI, shift[rank], numberOfElements[rank]);
        for (int i = 0; i < numberOfElements[rank]; i++){
            y_MPI[i] = AX_MPI[i] - B_MPI[i]; // y(n+1)
        }
        MPI_Allgatherv(y_MPI, numberOfElements[rank], MPI_DOUBLE, y, numberOfElements, shift, MPI_DOUBLE, MPI_COMM_WORLD);

        calculateMatrixOnVector(A_MPI, y, AY_MPI, shift[rank], numberOfElements[rank]);


        for (int i = 0; i < numberOfElements[rank]; ++i) {
            numerator += y_MPI[i ] * AY_MPI[i];
            denominator += AY_MPI[i] * AY_MPI[i];
        }

        MPI_Allreduce(&numerator, &numeratorSum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&denominator, &denominatorSum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        tetta = numeratorSum / denominatorSum;


        for (int i = 0; i < numberOfElements[rank]; ++i) {

            X_MPI[i] -= tetta * y_MPI[i]; // x(n+1)
        }
        MPI_Allgatherv(X_MPI, numberOfElements[rank], MPI_DOUBLE, x, numberOfElements, shift, MPI_DOUBLE, MPI_COMM_WORLD);

        condition = norm(y) / normB;
    }
    if (rank == 0) {
        printf("Execution time %lf \n", MPI_Wtime() - startTime);
        std::cout << "X = ";
        printVector(x, SIZE);
    }
    free(B_MPI);
    free(X_MPI);
    free(A_MPI);
    free(AX_MPI);
    free(AY_MPI);
    free(y);
    free(y_MPI);
    free(numberOfElements);
    free(shift);

    MPI_Finalize();
    return 0;
}
