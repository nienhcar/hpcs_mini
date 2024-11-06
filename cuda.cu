#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define MAX_PATTERN_LENGTH 64
#define ALPHABET_SIZE 256
#define WARP_SIZE 32
#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 8
#define CHUNK_SIZE 256

__constant__ unsigned long long d_pattern_mask[ALPHABET_SIZE];

__global__ void bitap_search_kernel(const unsigned char *text, size_t text_length, int pattern_length, int *results) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t start = (y * gridDim.x + x) * CHUNK_SIZE;
    size_t end = min(start + CHUNK_SIZE, text_length);

    __shared__ unsigned long long s_pattern_mask[ALPHABET_SIZE];
    for (int i = threadIdx.x; i < ALPHABET_SIZE; i += blockDim.x) {
        s_pattern_mask[i] = d_pattern_mask[i];
    }
    __syncthreads();

    unsigned long long R = ~0ULL;
    unsigned long long match = 1ULL << (pattern_length - 1);

    for (size_t i = start; i < end; ++i) {
        R = (R << 1) | s_pattern_mask[(unsigned char)text[i]];
        if ((R & match) == 0) {
            results[i - pattern_length + 1] = 1; // Adjusted the position
        }
    }
}

void cuda_bitap_search(const unsigned char *h_text, size_t text_length, const char *pattern, FILE *output_file) {
    int pattern_length = strlen(pattern);
    if (pattern_length == 0 || pattern_length > MAX_PATTERN_LENGTH) {
        fprintf(output_file, "Pattern is empty or too long!\n");
        return;
    }

    unsigned long long h_pattern_mask[ALPHABET_SIZE];
    for (int i = 0; i < ALPHABET_SIZE; ++i)
        h_pattern_mask[i] = ~0ULL;
    for (int i = 0; i < pattern_length; ++i)
        h_pattern_mask[(unsigned char)pattern[i]] &= ~(1ULL << i);

    cudaMemcpyToSymbol(d_pattern_mask, h_pattern_mask, ALPHABET_SIZE * sizeof(unsigned long long));

    unsigned char *d_text;
    int *d_results;
    cudaMalloc((void**)&d_text, text_length * sizeof(unsigned char));
    cudaMalloc((void**)&d_results, text_length * sizeof(int));
    cudaMemcpy(d_text, h_text, text_length * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemset(d_results, 0, text_length * sizeof(int));

    dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridSize((text_length + CHUNK_SIZE * BLOCK_SIZE_X - 1) / (CHUNK_SIZE * BLOCK_SIZE_X),
                  (text_length + CHUNK_SIZE * BLOCK_SIZE_Y - 1) / (CHUNK_SIZE * BLOCK_SIZE_Y));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    bitap_search_kernel<<<gridSize, blockSize>>>(d_text, text_length, pattern_length, d_results);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    int *h_results = (int*)malloc(text_length * sizeof(int));
    cudaMemcpy(h_results, d_results, text_length * sizeof(int), cudaMemcpyDeviceToHost);

    int found = 0;
    for (size_t i = 0; i < text_length; ++i) {
        if (h_results[i]) {
            fprintf(output_file, "Pattern found at position: %zu\n", i);
            found = 1;
        }
    }

    if (!found) {
        fprintf(output_file, "No match found.\n");
    }

    printf("Time taken: %f seconds\n", milliseconds / 1000.0);

    cudaFree(d_text);
    cudaFree(d_results);
    free(h_results);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    const char *pattern = "AGGA";
    unsigned char *text;
    size_t file_size;

    FILE *file = fopen("/content/drive/MyDrive/input_1L.txt", "rb");
    if (file == NULL) {
        perror("Could not open input.txt");
        return EXIT_FAILURE;
    }

    fseek(file, 0, SEEK_END);
    file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    text = (unsigned char*)malloc(file_size);
    if (text == NULL) {
        perror("Memory allocation failed");
        fclose(file);
        return EXIT_FAILURE;
    }

    size_t bytes_read = fread(text, 1, file_size, file);
    if (bytes_read != file_size) {
        perror("Error reading file");
        free(text);
        fclose(file);
        return EXIT_FAILURE;
    }
    fclose(file);

    FILE *output_file = fopen("output_1Lnew.txt", "w");
    if (output_file == NULL) {
        perror("Could not open output.txt");
        free(text);
        return EXIT_FAILURE;
    }

    cuda_bitap_search(text, file_size, pattern, output_file);

    free(text);
    fclose(output_file);

    return 0;
}
