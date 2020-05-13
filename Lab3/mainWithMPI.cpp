#include <cstdlib>
#include <mpi.h>
#include <iostream>

#define M 2000
#define N 2000
#define K 2000
#define GRID_DIMS 2


void print_matrix(double *matrix, int rows, int cols){
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix[i*cols + j] << ' ';
        }
        std::cout << std::endl;
    }
}

template <class T>
void print_vector(T * arr, int size){
    for (int i = 0; i < size; ++i){
        std::cout << arr[i] << ' ';
    }
    std::cout << std::endl;
}

void fill(double *matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i * cols + j] = 1;
        }
    }
}

void partMultiply(double *c, const double *a, const double *b, int sizeRow, int sizeColumn) {
    for (int i = 0; i < sizeRow; ++i) {
        for (int j = 0; j < sizeColumn; ++j) {
            c[i * sizeColumn + j] = 0;
            for (int k = 0; k < N; ++k) {
                c[i * sizeColumn + j] += a[i * N + k] * b[k * sizeColumn + j];
            }
        }
    }
}

void create1DComms(MPI_Comm grid2D, MPI_Comm *columns, MPI_Comm *rows) {
    int rowsFlags[2]={0, 1};
    int columnsFlags[2]={1, 0};
    MPI_Cart_sub(grid2D, columnsFlags, columns);
    MPI_Cart_sub(grid2D, rowsFlags, rows);
}

void createTypes(MPI_Datatype *BType, MPI_Datatype *CType, int sizeRow, int sizeColumn) {
    MPI_Type_vector(N, sizeColumn, K, MPI_DOUBLE, BType);
    MPI_Type_vector(sizeRow, sizeColumn, K, MPI_DOUBLE, CType);
    MPI_Type_create_resized(*BType, 0, sizeColumn * sizeof(double), BType);
    MPI_Type_create_resized(*CType, 0, sizeColumn * sizeof(double), CType);
    MPI_Type_commit(BType);
    MPI_Type_commit(CType);
}

void fillBArrays(int *elemsNumB, int *shiftsB, const int *dims) {
    for (int i = 0; i < dims[1]; ++i) {
        shiftsB[i] = i;
        elemsNumB[i] = 1;
    }
}

void fillCArrays(int *elemsNumC, int *shiftsC, int sizeRow, const int *dims, int grid2DSize) {
    for (int i = 0; i < grid2DSize; ++i) {
        elemsNumC[i] = 1;
    }
    for (int i = 0; i < dims[0]; ++i) {
        for (int j = 0; j < dims[1]; ++j) {
            shiftsC[i * dims[1] + j] = i * dims[1] * sizeRow + j;
        }
    }
}


void calculate(double *A, double *B, double *C, int *dims, int rank, MPI_Comm grid2D) {
    int sizeRow = M / dims[0];
    int sizeColumn = K / dims[1];
    int grid2DSize = 0;
    MPI_Comm_size(grid2D, &grid2DSize);
    int coordinatesOfProc[2];
    MPI_Cart_coords(grid2D, rank, GRID_DIMS, coordinatesOfProc);

    double *partA = new double[sizeRow * N];
    double *partB = new double[sizeColumn * N];
    double *partC = new double[sizeColumn * sizeRow];

    int *elemsNumB = nullptr;
    int *shiftsB = nullptr;
    int *elemsNumC = nullptr;
    int *shiftsC = nullptr;
    MPI_Datatype BType, CType;
    if (rank == 0) {
        elemsNumB = new int[dims[1]];
        shiftsB = new int[dims[1]];
        elemsNumC = new int[grid2DSize];
        shiftsC = new int[grid2DSize];
        createTypes(&BType, &CType, sizeRow, sizeColumn);
        fillBArrays(elemsNumB, shiftsB, dims);
        fillCArrays(elemsNumC, shiftsC, sizeRow, dims, grid2DSize);
    }
    MPI_Comm columns1D;
    MPI_Comm rows1D;
    create1DComms(grid2D, &columns1D, &rows1D);
    if (coordinatesOfProc[1] == 0) {
        MPI_Scatter(A, sizeRow * N, MPI_DOUBLE, partA, sizeRow * N, MPI_DOUBLE, 0, columns1D);
    }
    if (coordinatesOfProc[0] == 0) {
        MPI_Scatterv(B, elemsNumB, shiftsB, BType, partB, sizeColumn * N, MPI_DOUBLE, 0, rows1D);
    }
    MPI_Bcast(partA, sizeRow * N, MPI_DOUBLE, 0, rows1D);
    MPI_Bcast(partB, sizeColumn * N, MPI_DOUBLE, 0, columns1D);

    partMultiply(partC, partA, partB, sizeRow, sizeColumn);

    MPI_Gatherv(partC, sizeColumn * sizeRow, MPI_DOUBLE, C, elemsNumC, shiftsC, CType, 0, grid2D);
    if (rank == 0) {
        delete [] elemsNumB;
        delete [] shiftsB;
        delete [] elemsNumC;
        delete [] shiftsC;
        MPI_Type_free(&BType);
        MPI_Type_free(&CType);
    }
    delete [] partA;
    delete [] partB;
    delete [] partC;
}


int main(int argc, char *argv[]) {
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int dims[2] = {0};
    int periods[2] = {0};
    int reorder = 0;
    MPI_Comm grid2D;
    MPI_Dims_create(size, GRID_DIMS, dims);
    MPI_Cart_create(MPI_COMM_WORLD, GRID_DIMS, dims, periods, reorder, &grid2D);

    double *A = nullptr;
    double *B = nullptr;
    double *C = nullptr;

    if (rank == 0) {
        A = new double[M * N];
        B = new double[K * N];
        C = new double[K * M];
        fill(A, M, N);
        fill(B, N, K);
    }
    double startTime = MPI_Wtime();

    calculate(A, B, C, dims, rank, grid2D);
    if(rank==0){
        double endTime = MPI_Wtime();
        //print_matrix(C, M, K);
        std::cout << "Time = " << endTime - startTime << std::endl;
    }
    if(rank==0){
        delete [] A;
        delete [] B;
        delete [] C;
    }
    MPI_Finalize();
    return 0;
}