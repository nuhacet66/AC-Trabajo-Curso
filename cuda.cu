#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <string.h>

#define MATRIX_SIZE 2056
#define BLOCK_SIZE 32
#define CLOCKS_PER_SEC 1000000

__global__ void multiply_matrices(int *matrix_a, int *matrix_b, int *matrix_result) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < MATRIX_SIZE && col < MATRIX_SIZE) {
        int sum = 0;
        for (int i = 0; i < MATRIX_SIZE; i++) {
            sum += matrix_a[row * MATRIX_SIZE + i] * matrix_b[i * MATRIX_SIZE + col];
        }
        matrix_result[row * MATRIX_SIZE + col] = sum;
    }
}

void print_matrix(int *matrix) {
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
        printf("%d ", matrix[i]);
        if ((i + 1) % MATRIX_SIZE == 0) {
            printf("\n");
        }
    }
    printf("\n");
}

int main() {
    int *host_matrix_a, *host_matrix_b, *host_matrix_result;
    int *device_matrix_a, *device_matrix_b, *device_matrix_result;
    size_t size = MATRIX_SIZE * MATRIX_SIZE * sizeof(int);
    srand(time(NULL));

    // Allocate memory on host
    host_matrix_a = (int *)malloc(size);
    host_matrix_b = (int *)malloc(size);
    host_matrix_result = (int *)malloc(size);

    // Initialize matrices on host
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
        host_matrix_a[i] = rand() % 10;
        host_matrix_b[i] = rand() % 10;
    }

    printf("Matrix A:\n");
    print_matrix(host_matrix_a);
    printf("Matrix B:\n");
    print_matrix(host_matrix_b);

    // Allocate memory on device
    cudaMalloc(&device_matrix_a, size);
    cudaMalloc(&device_matrix_b, size);
    cudaMalloc(&device_matrix_result, size);

    // Copy matrices from host to device
    cudaMemcpy(device_matrix_a, host_matrix_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_matrix_b, host_matrix_b, size, cudaMemcpyHostToDevice);

    // Initialize result matrix on host
    memset(host_matrix_result, 0, size);

    // Define block and grid sizes
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks(MATRIX_SIZE / threadsPerBlock.x, MATRIX_SIZE / threadsPerBlock.y);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start recording events
    cudaEventRecord(start);

    // Launch kernel
    multiply_matrices<<<numBlocks, threadsPerBlock>>>(device_matrix_a, device_matrix_b, device_matrix_result);

    // Stop recording events
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Check for errors during kernel execution
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(error));
    }

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy result matrix from device to host
    cudaMemcpy(host_matrix_result, device_matrix_result, size, cudaMemcpyDeviceToHost);

    // Print result matrix
    printf("Matrix Result:\n");
    print_matrix(host_matrix_result);

    // Free host and device memory
    free(host_matrix_a);
    free(host_matrix_b);
    free(host_matrix_result);
    cudaFree(device_matrix_a);
    cudaFree(device_matrix_b);
    cudaFree(device_matrix_result);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Print execution time
    printf("\n-------------------\n");
    printf("Execution Time (GPU): %f milliseconds\n", milliseconds);

    return 0;
}
