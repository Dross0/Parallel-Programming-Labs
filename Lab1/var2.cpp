#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <mpi.h>
#include <iostream>

#define SIZE 35000
#define E 0.000001
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

    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < numberOfElem; j++) {
            matrixMpi[i * numberOfElem + j ] = 1.0;
            if (shift + j == i)
                ++matrixMpi[i * numberOfElem + j];
        }

    }
    for (int i = 0; i < numberOfElem; ++i) {
        b[i] = SIZE + 1;
        x[i] = rand() % 15;
    }

}


void calculateMatrixOnVector(double *A_MPI, double *v, double *res, int shift, int numberOfElem) {
    for (int i = 0; i < SIZE; i++){
        res[i] = scalar(&(A_MPI[i * numberOfElem]) , v, numberOfElem);
    }
}

void printVector(double * v, int size){
    for (int i = 0; i < size; ++i){
        printf("%f ", v[i]);
    }
    printf("\n");
}


void printMatrix(double * v, int size){
    for (int i = 0; i < size; ++i){
        printVector(v + i * size, size);
    }

}

int main(int argc, char *argv[]) {
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);//получение числа процессов
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);//получение номера процесса
    double startTime = MPI_Wtime();

    int *numberOfElements = (int *) calloc(size, sizeof(int));//количество элементов в процессе

    int *shift = (int *) calloc(size, sizeof(int));//сдвиг

    for (int i = 0; i < size; ++i) {
        numberOfElements[i] = (SIZE / size) + ((i < SIZE % size) ? (1) : (0));
    }
    for (int i = 1; i < size; ++i) {
        shift[i] = shift[i - 1] + numberOfElements[i - 1];
    }

    double * B_MPI = getArray(numberOfElements[rank]);
    double *X_MPI = getArray(numberOfElements[rank]);
    double *A_MPI = getArray(SIZE * numberOfElements[rank]);
    double *y_MPI = getArray(numberOfElements[rank]);
    double * AX_MPI = getArray(SIZE);
    double * AX = getArray(SIZE);
    double * AY = getArray(SIZE);
    double *AY_MPI = getArray(SIZE);
    fillTestDataMPI(A_MPI, B_MPI, X_MPI, shift[rank], numberOfElements[rank]);




    double b_MPI_norm = scalar(B_MPI, B_MPI, numberOfElements[rank]);
    double normB = 0;
    MPI_Allreduce(&b_MPI_norm, &normB, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    normB = sqrt(normB);
    double tetta = 0;
    double numerator = 0;
    double denominator = 0;
    double denominatorSum = 0;
    double numeratorSum = 0;
    double condition = 100;
    int iterations = 0;

    while (condition > E && iterations++ < MAX_ITERATIONS) {
        numerator = 0;
        numeratorSum = 0;
        denominator = 0;
        denominatorSum = 0;
        calculateMatrixOnVector(A_MPI, X_MPI, AX_MPI, shift[rank], numberOfElements[rank]);
        MPI_Allreduce(AX_MPI, AX, SIZE, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        for (int i = 0; i < numberOfElements[rank]; i++){
            y_MPI[i] = AX[i + shift[rank]] - B_MPI[i]; // y(n+1)
        }


        calculateMatrixOnVector(A_MPI, y_MPI, AY_MPI, shift[rank], numberOfElements[rank]);
        MPI_Allreduce(AY_MPI, AY, SIZE, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        for (int i = 0; i < numberOfElements[rank]; ++i) {
            numerator += y_MPI[i ] * AY[i + shift[rank]];
            denominator += AY[i + shift[rank]] * AY[i + shift[rank]];
        }

        MPI_Allreduce(&numerator, &numeratorSum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&denominator, &denominatorSum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        tetta = numeratorSum / denominatorSum;


        for (int i = 0; i < numberOfElements[rank]; ++i) {

            X_MPI[i] -= tetta * y_MPI[i]; // x(n+1)
        }

        double y_MPI_norm = scalar(y_MPI, y_MPI, numberOfElements[rank]);

        double y_norm = 0;
        MPI_Allreduce(&y_MPI_norm, &y_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        y_norm = sqrt(y_norm);


        condition = y_norm / normB;
    }
    double x[SIZE] = {0};
    MPI_Allgatherv(X_MPI, numberOfElements[rank], MPI_DOUBLE,x, numberOfElements, shift, MPI_DOUBLE, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Execution time %lf \n", MPI_Wtime() - startTime);
        printVector(x, SIZE);
    }
    free(B_MPI);
    free(X_MPI);
    free(A_MPI);
    free(AX_MPI);
    free(AY_MPI);
    free(y_MPI);
    free(numberOfElements);
    free(shift);

    MPI_Finalize();
    return 0;
}
