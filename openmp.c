#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define MATRIX_SIZE 800
#define CLOCKS_PER_SEC 1000000

void multiply_matrices(int **matrix_a, int **matrix_b, int **matrix_result) {
    int i, j, k, tid, nthreads;
    #pragma omp parallel shared(matrix_a, matrix_b, matrix_result, nthreads) private(tid, i, j, k)
        tid = omp_get_thread_num();
        #pragma omp for
        for (i = 0; i < MATRIX_SIZE; i++) {
            for (j = 0; j < MATRIX_SIZE; j++){
                matrix_a[i][j] = rand() % 10;
            }
        }
        #pragma omp for
        for (i = 0; i < MATRIX_SIZE; i++) {
            for (j = 0; j < MATRIX_SIZE; j++){
                matrix_b[i][j] = rand() % 10;
            }
        }
        #pragma omp for
        for (i = 0; i < MATRIX_SIZE; i++) {
            for (j = 0; j < MATRIX_SIZE; j++){
                matrix_result[i][j] = 0;
            }
        }
        #pragma omp for
        for (i = 0; i < MATRIX_SIZE; i++) {
            for (j = 0; j < MATRIX_SIZE; j++) {
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

int main(int argc, char *argv[]) {
    int **matrix_a, **matrix_b, **matrix_result;
    int i, j;
    srand(time(NULL));

    clock_t start = clock();

    matrix_a = (int **) malloc(MATRIX_SIZE * sizeof(int *));
    matrix_b = (int **) malloc(MATRIX_SIZE * sizeof(int *));
    matrix_result = (int **) malloc(MATRIX_SIZE * sizeof(int *));
    for (i = 0; i < MATRIX_SIZE; i++) {
        matrix_a[i] = (int *) malloc(MATRIX_SIZE * sizeof(int));
        matrix_b[i] = (int *) malloc(MATRIX_SIZE * sizeof(int));
        matrix_result[i] = (int *) malloc(MATRIX_SIZE * sizeof(int));
    }

    multiply_matrices(matrix_a, matrix_b, matrix_result);

    printf("\nMatrix A:\n");
    print_matrix(matrix_a);
    printf("\nMatrix B:\n");
    print_matrix(matrix_b);
    printf("\nMatrix Result (C):\n");
    print_matrix(matrix_result);

    for (i = 0; i < MATRIX_SIZE; i++) {
        free(matrix_a[i]);
        free(matrix_b[i]);
        free(matrix_result[i]);
    }
    free(matrix_a);
    free(matrix_b);
    free(matrix_result);

    printf("\n-------------------\n");
    printf("Tiempo de ejecución del programa (CPU): %f segundos\n", ((double)clock() - start) / CLOCKS_PER_SEC);

    return 0;
}
