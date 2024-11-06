#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_PATTERN_LENGTH 64
#define ALPHABET_SIZE 256

void bitap_search(const char *text, const char *pattern, FILE *output_file) {
    int m = strlen(pattern);
    long pattern_mask[ALPHABET_SIZE];
    long R = ~1;  // Initialize R
    int found = 0;  // Flag to check if any matches were found

    if (m == 0) {
        fprintf(output_file, "Pattern is empty!\n");
        return;
    }

    if (m > MAX_PATTERN_LENGTH) {
        fprintf(output_file, "Pattern is too long!\n");
        return;
    }

    // Initialize the pattern bitmasks
    for (int i = 0; i < ALPHABET_SIZE; ++i)
        pattern_mask[i] = ~0;

    for (int i = 0; i < m; ++i)
        pattern_mask[(unsigned char)pattern[i]] &= ~(1L << i);

    // Search through the text
    for (int i = 0; text[i]; ++i) {
        // Update the bit array
        R |= pattern_mask[(unsigned char)text[i]];
        R <<= 1;

        // Check for a match
        if ((R & (1L << m)) == 0) {
            fprintf(output_file, "Pattern found at position: %d\n", i - m + 1);
            found = 1;
        }
    }

    if (!found) {
        fprintf(output_file, "\nNo Match\n");
    }
}

int main() {
    const char *pattern = "AGGA";  // Your desired pattern
    char *text;                    // Pointer for the text
    long file_size;

    // Open input.txt to read
    FILE *file = fopen("input_1L.txt", "r");
    if (file == NULL) {
        perror("Could not open input.txt");
        return EXIT_FAILURE;
    }

    // Seek to the end of the file to get its size
    fseek(file, 0, SEEK_END);
    file_size = ftell(file);
    fseek(file, 0, SEEK_SET);  // Reset the file pointer to the beginning

    // Allocate memory for the text based on the file size
    text = malloc(file_size + 1);  // +1 for the null terminator
    if (text == NULL) {
        perror("Memory allocation failed");
        fclose(file);
        return EXIT_FAILURE;
    }

    // Read the entire file into the text buffer
    fread(text, sizeof(char), file_size, file);
    fclose(file);
    text[file_size] = '\0';  // Ensure null-termination

    // Open the output file for writing
    FILE *output_file = fopen("outputseq.txt", "w");
    if (output_file == NULL) {
        perror("Could not open outputseq.txt");
        free(text);
        return EXIT_FAILURE;
    }

    // Start time measurement
    clock_t start_time = clock();

    // Search for the pattern and write results to the output file
    bitap_search(text, pattern, output_file);

    // End time measurement
    clock_t end_time = clock();

    // Calculate and print execution time to the console
    double time_taken = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("\nTime taken: %f seconds\n", time_taken);

    // Clean up
    free(text);
    fclose(output_file);

    return 0;
}
