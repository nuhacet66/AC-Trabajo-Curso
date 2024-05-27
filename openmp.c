#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <papi.h>

#define MATRIX_SIZE 800

void handle_error(int retval) {
    printf("PAPI error %d: %s\n", retval, PAPI_strerror(retval));
    exit(1);
}

void multiply_matrices(int **matrix_a, int **matrix_b, int **matrix_result) {
    int i, j, k, tid, nthreads;
    #pragma omp parallel shared(matrix_a, matrix_b, matrix_result, nthreads) private(tid, i, j, k)
    {
        tid = omp_get_thread_num();
        #pragma omp for
        for (i = 0; i < MATRIX_SIZE; i++) {
            for (j = 0; j < MATRIX_SIZE; j++) {
                matrix_a[i][j] = rand() % 10;
            }
        }
        #pragma omp for
        for (i = 0; i < MATRIX_SIZE; i++) {
            for (j = 0; j < MATRIX_SIZE; j++) {
                matrix_b[i][j] = rand() % 10;
            }
        }
        #pragma omp for
        for (i = 0; i < MATRIX_SIZE; i++) {
            for (j = 0; j < MATRIX_SIZE; j++) {
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

    int retval;
    long long values[5];
    int events[5] = {PAPI_TOT_CYC, PAPI_TOT_INS, PAPI_FP_INS, PAPI_LD_INS, PAPI_SR_INS};

    // Inicializar PAPI
    if ((retval = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT) {
        fprintf(stderr, "PAPI library init error!\n");
        exit(1);
    }

    // Crear un EventSet
    int EventSet = PAPI_NULL;
    if ((retval = PAPI_create_eventset(&EventSet)) != PAPI_OK) {
        handle_error(retval);
    }

    // Agregar eventos al EventSet
    if ((retval = PAPI_add_events(EventSet, events, 5)) != PAPI_OK) {
        handle_error(retval);
    }

    // Iniciar el conteo
    if ((retval = PAPI_start(EventSet)) != PAPI_OK) {
        handle_error(retval);
    }

    // Inicializar matrices
    matrix_a = (int **)malloc(MATRIX_SIZE * sizeof(int *));
    matrix_b = (int **)malloc(MATRIX_SIZE * sizeof(int *));
    matrix_result = (int **)malloc(MATRIX_SIZE * sizeof(int *));
    for (i = 0; i < MATRIX_SIZE; i++) {
        matrix_a[i] = (int *)malloc(MATRIX_SIZE * sizeof(int));
        matrix_b[i] = (int *)malloc(MATRIX_SIZE * sizeof(int));
        matrix_result[i] = (int *)malloc(MATRIX_SIZE * sizeof(int));
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

    // Detener el conteo
    if ((retval = PAPI_stop(EventSet, values)) != PAPI_OK) {
        handle_error(retval);
    }

    // Imprimir los resultados
    printf("\n-------------------\n");
    printf("Total cycles: %lld\n", values[0]);
    printf("Total instructions: %lld\n", values[1]);
    printf("Floating point instructions: %lld\n", values[2]);
    printf("Load instructions: %lld\n", values[3]);
    printf("Store instructions: %lld\n", values[4]);

    // Limpiar PAPI
    if ((retval = PAPI_cleanup_eventset(EventSet)) != PAPI_OK) {
        handle_error(retval);
    }

    if ((retval = PAPI_destroy_eventset(&EventSet)) != PAPI_OK) {
        handle_error(retval);
    }

    PAPI_shutdown();

    return 0;
}
