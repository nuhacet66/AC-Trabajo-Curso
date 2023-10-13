#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#define MATRIX_SIZE 6
#define CLOCKS_PER_SEC 1000000

void multiply_matrices(int **matrix_a, int **matrix_b, int **matrix_result) {
    int i, j, k;
    for (i = 0; i < MATRIX_SIZE; i++) {
        for (j = 0; j < MATRIX_SIZE; j++) {
            matrix_result[i][j] = 0;
            for (k = 0; k < MATRIX_SIZE; k++) {
                matrix_result[i][j] += matrix_a[i][k] * matrix_b[k][j];
            }
        }
    }
}

void print_matrix(int **matrix) {
    int i, j;
    for (i = 0; i < MATRIX_SIZE; i++) {
        for (j = 0; j < MATRIX_SIZE; j++) {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

void fill_matrix(int **matrix) {
    int i, j;
    for (i = 0; i < MATRIX_SIZE; i++) {
        matrix[i] = (int *)malloc(MATRIX_SIZE * sizeof(int));
        for (j = 0; j < MATRIX_SIZE; j++) {
            matrix[i][j] = rand() % 10;
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    clock_t start = clock();

    if (size != MATRIX_SIZE) {
        if (rank == 0) {
            printf("Este programa está diseñado para funcionar con %d procesos MPI.\n", MATRIX_SIZE);
        }
        MPI_Finalize();
        return 1;
    }

    int **matrix_a, **matrix_b, **matrix_c;
    srand(time(NULL));

    if (rank == 0) {
        fill_matrix(matrix_a);
        fill_matrix(matrix_b);

        printf("\nMatrix A:\n");
        print_matrix(matrix_a);

        printf("\nMatrix B:\n");
        print_matrix(matrix_b);
    }

    MPI_Bcast(&matrix_a[0][0], MATRIX_SIZE * MATRIX_SIZE, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&matrix_b[0][0], MATRIX_SIZE * MATRIX_SIZE, MPI_INT, 0, MPI_COMM_WORLD);

    multiply_matrices(matrix_a, matrix_b, matrix_c);

    MPI_Gather(&matrix_c[0][0], MATRIX_SIZE * MATRIX_SIZE, MPI_INT, &matrix_c[0][0], MATRIX_SIZE * MATRIX_SIZE, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\nMatrix C (resultado):\n");
        print_matrix(matrix_c);
    }

    printf("\n-------------------\n");
    printf("Proceso %d de %d\n", rank, size);
    printf("Tiempo de ejecución del programa (CPU): %f segundos\n", ((double)clock() - start) / CLOCKS_PER_SEC);

    MPI_Finalize();

    return 0;
}
