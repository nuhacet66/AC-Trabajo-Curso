#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <papi.h>

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
    int retval;
    int EventSet = PAPI_NULL;
    long long values[5];

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Initialize PAPI
    if ((retval = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT) {
        fprintf(stderr, "Error: PAPI library initialization failed!\n");
        MPI_Finalize();
        return 1;
    }
    
    if ((retval = PAPI_create_eventset(&EventSet)) != PAPI_OK) {
        fprintf(stderr, "Error: PAPI create event set failed!\n");
        MPI_Finalize();
        return 1;
    }
    
    if ((retval = PAPI_add_event(EventSet, PAPI_TOT_CYC)) != PAPI_OK ||
        (retval = PAPI_add_event(EventSet, PAPI_TOT_INS)) != PAPI_OK ||
        (retval = PAPI_add_event(EventSet, PAPI_FP_INS)) != PAPI_OK ||
        (retval = PAPI_add_event(EventSet, PAPI_LD_INS)) != PAPI_OK ||
        (retval = PAPI_add_event(EventSet, PAPI_SR_INS)) != PAPI_OK) {
        fprintf(stderr, "Error: PAPI add event failed!\n");
        MPI_Finalize();
        return 1;
    }
    
    if ((retval = PAPI_start(EventSet)) != PAPI_OK) {
        fprintf(stderr, "Error: PAPI start counters failed!\n");
        MPI_Finalize();
        return 1;
    }

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
        matrix_a = (int **)malloc(MATRIX_SIZE * sizeof(int *));
        matrix_b = (int **)malloc(MATRIX_SIZE * sizeof(int *));
        matrix_c = (int **)malloc(MATRIX_SIZE * sizeof(int *));
        for (int i = 0; i < MATRIX_SIZE; i++) {
            matrix_a[i] = (int *)malloc(MATRIX_SIZE * sizeof(int));
            matrix_b[i] = (int *)malloc(MATRIX_SIZE * sizeof(int));
            matrix_c[i] = (int *)malloc(MATRIX_SIZE * sizeof(int));
        }

        fill_matrix(matrix_a);
        fill_matrix(matrix_b);

        printf("\nMatrix A:\n");
        print_matrix(matrix_a);

        printf("\nMatrix B:\n");
        print_matrix(matrix_b);
    }

    // Allocate memory for matrices in all processes
    if (rank != 0) {
        matrix_a = (int **)malloc(MATRIX_SIZE * sizeof(int *));
        matrix_b = (int **)malloc(MATRIX_SIZE * sizeof(int *));
        matrix_c = (int **)malloc(MATRIX_SIZE * sizeof(int *));
        for (int i = 0; i < MATRIX_SIZE; i++) {
            matrix_a[i] = (int *)malloc(MATRIX_SIZE * sizeof(int));
            matrix_b[i] = (int *)malloc(MATRIX_SIZE * sizeof(int));
            matrix_c[i] = (int *)malloc(MATRIX_SIZE * sizeof(int));
        }
    }

    // Broadcast matrices A and B to all processes
    for (int i = 0; i < MATRIX_SIZE; i++) {
        MPI_Bcast(matrix_a[i], MATRIX_SIZE, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(matrix_b[i], MATRIX_SIZE, MPI_INT, 0, MPI_COMM_WORLD);
    }

    multiply_matrices(matrix_a, matrix_b, matrix_c);

    // Gather results from all processes to process 0
    for (int i = 0; i < MATRIX_SIZE; i++) {
        MPI_Gather(matrix_c[i], MATRIX_SIZE, MPI_INT, matrix_c[i], MATRIX_SIZE, MPI_INT, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        printf("\nMatrix C (resultado):\n");
        print_matrix(matrix_c);
    }

    if ((retval = PAPI_stop(EventSet, values)) != PAPI_OK) {
        fprintf(stderr, "Error: PAPI stop counters failed!\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("\n-------------------\n");
        printf("PAPI_TOT_CYC: %lld\n", values[0]);
        printf("PAPI_TOT_INS: %lld\n", values[1]);
        printf("PAPI_FP_INS: %lld\n", values[2]);
        printf("PAPI_LD_INS: %lld\n", values[3]);
        printf("PAPI_SR_INS: %lld\n", values[4]);
    }

    printf("Proceso %d de %d\n", rank, size);
    printf("Tiempo de ejecución del programa (CPU): %f segundos\n", ((double)clock() - start) / CLOCKS_PER_SEC);

    MPI_Finalize();

    return 0;
}
