#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <algorithm>

#define MAX_PATTERN_LENGTH 64
#define ALPHABET_SIZE 256
#define MAX_RESULTS 1048576

typedef struct {
    int position;
} Match;

__constant__ unsigned long long d_pattern_mask[ALPHABET_SIZE];

__global__ void bitap_search_kernel(const unsigned char* text,
                                    const size_t text_length,
                                    const int pattern_length,
                                    int* match_positions,
                                    int* match_count) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = gridDim.x * blockDim.x;

    for (size_t start_pos = tid; start_pos < text_length - pattern_length + 1; start_pos += stride) {
        unsigned long long R = ~0ULL;

        #pragma unroll 4
        for (int i = 0; i < pattern_length; i++) {
            R = ((R << 1) | d_pattern_mask[text[start_pos + i]]);
        }

        if (!(R & (1ULL << (pattern_length - 1)))) {
            int index = atomicAdd(match_count, 1);
            if (index < MAX_RESULTS) {
                match_positions[index] = start_pos;
            }
        }
    }
}

void cuda_bitap_search(const unsigned char *h_text, size_t text_length, const char *pattern, FILE *output_file) {
    cudaDeviceProp props;
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaGetDeviceProperties(&props, deviceId);

    const int block_size = 128;
    const int sm_count = props.multiProcessorCount;
    const int blocks_per_sm = 8;
    const int num_blocks = sm_count * blocks_per_sm;

    int pattern_length = strlen(pattern);
    if (pattern_length == 0 || pattern_length > MAX_PATTERN_LENGTH) {
        fprintf(output_file, "Pattern is empty or too long!\n");
        return;
    }

    unsigned long long h_pattern_mask[ALPHABET_SIZE];
    memset(h_pattern_mask, 0xFF, sizeof(h_pattern_mask));

    for (int i = 0; i < pattern_length; i++) {
        h_pattern_mask[(unsigned char)pattern[i]] &= ~(1ULL << i);
    }

    cudaMemcpyToSymbol(d_pattern_mask, h_pattern_mask, sizeof(h_pattern_mask));

    unsigned char *d_text;
    int *d_match_positions, *d_match_count;

    cudaMalloc(&d_text, text_length);
    cudaMalloc(&d_match_positions, MAX_RESULTS * sizeof(int));
    cudaMalloc(&d_match_count, sizeof(int));

    cudaMemcpy(d_text, h_text, text_length, cudaMemcpyHostToDevice);
    cudaMemset(d_match_count, 0, sizeof(int));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    bitap_search_kernel<<<num_blocks, block_size>>>(
        d_text, text_length, pattern_length, d_match_positions, d_match_count
    );
    cudaEventRecord(stop);

    float milliseconds = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    int h_match_count;
    int h_match_positions[MAX_RESULTS];
    cudaMemcpy(&h_match_count, d_match_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_match_positions, d_match_positions, h_match_count * sizeof(int), cudaMemcpyDeviceToHost);

    // Sort the match positions
    std::sort(h_match_positions, h_match_positions + h_match_count);

    fprintf(output_file, "Found %d matches at positions:\n", h_match_count);
    for (int i = 0; i < h_match_count && i < MAX_RESULTS; i++) {
        fprintf(output_file, "%d\n", h_match_positions[i]);
    }

    printf("Time taken: %f milliseconds\n", milliseconds);

    cudaFree(d_text);
    cudaFree(d_match_positions);
    cudaFree(d_match_count);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    const char *pattern = "AGGA";
    unsigned char *text;
    size_t file_size;

    FILE *file = fopen("/content/drive/MyDrive/input_1L.txt", "rb");
    if (!file) {
        perror("Could not open input.txt");
        return EXIT_FAILURE;
    }

    fseek(file, 0, SEEK_END);
    file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    text = (unsigned char*)malloc(file_size);
    if (!text) {
        perror("Memory allocation failed");
        fclose(file);
        return EXIT_FAILURE;
    }

    if (fread(text, 1, file_size, file) != file_size) {
        perror("Error reading file");
        free(text);
        fclose(file);
        return EXIT_FAILURE;
    }
    fclose(file);

    FILE *output_file = fopen("1Llo2.txt", "w");
    if (!output_file) {
        perror("Could not open output.txt");
        free(text);
        return EXIT_FAILURE;
    }

    cuda_bitap_search(text, file_size, pattern, output_file);

    free(text);
    fclose(output_file);

    return 0;
}
