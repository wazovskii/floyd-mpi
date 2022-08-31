#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>
#include <mpi.h>
#include <string.h>
#define INF INT_MAX
#define BUZZ_SIZE 1024


int getMatrixSize(char *inFile) {
    FILE *fp;
    char ch;
    int size = 0;
    if ((fp = fopen(inFile, "r"))==NULL) {
        printf("Cannot open file.\n");
        exit(1);
    }
    char str[1024];
    fgets(str, BUZZ_SIZE, fp);
    char *pch = strtok(str," ");
    while (pch != NULL){
        pch = strtok (NULL, " ");
        size++;
    }
    fclose(fp);
    return size;
}

int** parseFile(char *inFile, int size) {
    FILE *fp;
    char ch;
    int i = 0, j = 0;
    int **matrix = (int**)malloc(size * sizeof(int*));
    for (int i = 0; i < size; i++) {
        matrix[i] = (int*)malloc(size * sizeof(int));
    }
    if ((fp = fopen(inFile, "r"))==NULL) {
        printf("Cannot open file.\n");
        exit(1);
    }
    char str[1024];
    while (fgets(str, BUZZ_SIZE, fp)!=NULL){
        str[strcspn(str,"\n")] = 0;
        char *pch = strtok(str," ");
        
        while (pch != NULL){
            if (strcmp(pch, "~") == 0)
                matrix[i][j] = INF;
            else
                matrix[i][j] = atoi(pch);
            pch = strtok (NULL, " ");
            j++;
        }
        i++;
        j=0;
    }
    return matrix;
}
int** generateMatrix(int size) {
    FILE *fp;
    srand(time(NULL));
    int **matrix = (int**)malloc(size * sizeof(int*));
    for (int i = 0; i < size; i++) {
        matrix[i] = (int*)malloc(size * sizeof(int));
    }
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (i == j)
                matrix[i][j] = 0;
            else {
                matrix[i][j] = rand() % INT_MAX / 100;
                if (matrix[i][j] == 0) {
                    matrix[i][j] = INF;
                }
            }
        }
    }
    for (int k = 0; k < size*size/2; k++) {
        int i = rand () % size;
        int j = rand () % size;
        if (i != j)
            matrix[i][j] = INF;
    }
    if ((fp = fopen("matrixLog.txt", "w"))==NULL) {
        printf("Cannot open file.\n");
        exit(1);
    }
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            fprintf(fp, "%12d", matrix[i][j]);
        }
        fprintf(fp, "\n");
    }
    return matrix;
}

int min(int a, int b) {
    int result = b;
    if (a < b)
        result = a;
    return result;
}

int findRank(int k, int rows_per_proc, int size, int commsize) {
    int result = 0;
    int i = 0;
    while (k >= 0) {
        k -= rows_per_proc + (i < size % commsize);
        i++;
    }
    return i-1;
}

int rowInSubMatrix(int k, int rows_per_proc, int size, int commsize) {
    int rank = findRank(k, rows_per_proc, size, commsize);
    int i = 0;
    while (rank > 0) {
        k -= rows_per_proc + (i < size % commsize);
        i++;
        rank--;
    }
    return k;
}

int* floyd(int *matrix, int size) {
    int commsize, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &commsize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int *recMatrix;
    int *kLine = malloc(size*sizeof(int));
    int rows_per_proc = size / commsize;
    int ub = rows_per_proc + (rank < (size % commsize)) - 1;
    
    int *displs = (int *)malloc(commsize*sizeof(int));
    int *rcounts = (int *)malloc(commsize*sizeof(int));
    int *chunk_size = (int *)malloc(commsize*sizeof(int));
    if (rank == 0) {
        for (int i=0; i < commsize; ++i) {
            if (i)
                displs[i] = rcounts[i-1] + displs[i-1];
            else
                displs[i] = 0;
            rcounts[i] = rows_per_proc + (i < (size % commsize));
            rcounts[i] *= size;
            chunk_size[i] = rcounts[i];
        }
    }
    int rsize = (rows_per_proc + (rank < (size % commsize))) * size;
    recMatrix = malloc(rsize * sizeof(int));
    
    if (rank == 0) {
        for (int q = 0; q < size; q++) {
            kLine[q] = matrix[q];
        }
    }
    MPI_Scatterv(matrix, chunk_size, displs, MPI_INT, recMatrix, rsize, MPI_INT, 0, MPI_COMM_WORLD);
    for (int k = 0; k < size; k++) {
        MPI_Bcast(kLine, size, MPI_INT, findRank(k, rows_per_proc, size, commsize), MPI_COMM_WORLD);
        for (int i = 0; i <= ub; i++) {
            for (int j = 0; j < size; j++) {
                if (recMatrix[i * size + k] != INF && kLine[j] != INF && recMatrix[i*size+k] + kLine[j] < recMatrix[i*size+j]) {
                    recMatrix[i*size+j] = recMatrix[i*size+k] + kLine[j];
                }
            }
        }
        int nextK;
        if ((k + 1) >= size)
            nextK = k;
        else
            nextK = k+1;
        if (rank == findRank(nextK, rows_per_proc, size, commsize)) {
            int row_this_proc = rows_per_proc + (rank <= (size % commsize));
            
            for (int q = 0; q < size; q++) {
                kLine[q] = recMatrix[(rowInSubMatrix(nextK, rows_per_proc, size, commsize))*size+q];
            }
        }
    }
    MPI_Gatherv( recMatrix, rsize, MPI_INT, matrix, rcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);
    return matrix;
}



void checkAlgo(int *matrix, int size) {
    int error = 0;
    int trueMatrix5[5][5] = {
        {0, 1, -3, 2, -4},
        {3, 0, -4, 1, -1},
        {7, 4, 0, 5, 3},
        {2, -1, -5, 0, -2},
        {8, 5, 1, 6, 0}};
    int trueMatrix[4][4] = {
        {0, 3, 7, 5},
        {2, 0, 6, 4},
        {3, 1, 0, 5},
        {5, 3, 2, 0}};
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (matrix[i*size+j] != trueMatrix[i][j]) {
                error = 1;
                printf("lm [%d][%d] = %d rm [%d][%d] = %d\n", i, j, matrix[i*size+j], i, j, trueMatrix[i][j]);
            }
        }
    }

    if (error)
        printf("Error in algorithm\n");
    else
        printf("Algorithm tested\n");
}

int main(int argc, char* argv[]) {
    int **matrix;
    double cpu_time_used;
    int commsize, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &commsize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int size;
    
    int *dMatrix;
    size = 100;
    
    matrix = generateMatrix(size);
    dMatrix = malloc(size*size*sizeof(int));
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            dMatrix[i*size+j] = matrix[i][j];
        }
    }
    double start = MPI_Wtime();
    dMatrix = floyd(dMatrix, size);
    double stop = MPI_Wtime();
    double duration = stop - start;
    MPI_Finalize();
    if (rank == 0)
       printf("Runtime is %1.3f seconds\n", duration);
    return 0;
}
