#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define MAX_PATTERN_LENGTH 64
#define ALPHABET_SIZE 256
#define MAX_MATCHES 1000  // Maximum number of matches per process

void bitap_search(const char *text, int text_length, const char *pattern, int rank, int global_offset, int *local_matches, int *match_count) {
    int m = strlen(pattern);
    long pattern_mask[ALPHABET_SIZE];
    long R = ~1;  // Initialize R
    int local_found = 0;  // Flag to check if any matches were found locally

    if (m == 0 || m > MAX_PATTERN_LENGTH) {
        if (rank == 0) {
            printf("Pattern is empty or too long!\n");
        }
        return;
    }

    // Initialize the pattern bitmasks
    for (int i = 0; i < ALPHABET_SIZE; ++i)
        pattern_mask[i] = ~0;
    for (int i = 0; i < m; ++i)
        pattern_mask[(unsigned char)pattern[i]] &= ~(1L << i);

    // Search through the local portion of the text
    for (int i = 0; i < text_length; ++i) {
        // Update the bit array
        R |= pattern_mask[(unsigned char)text[i]];
        R <<= 1;

        // Check for a match
        if ((R & (1L << m)) == 0) {
            // Store the global position where the match was found
            int global_position = global_offset + i - m + 1; // Calculate the global match position
            if (local_found < MAX_MATCHES && global_position >= 0) { // Ensure position is non-negative
                local_matches[local_found] = global_position; // Store the match position
                local_found++;
            }
        }
    }

    *match_count = local_found; // Store the number of matches found
}

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const char *pattern = "AGGA";  // Your desired pattern
    char *text = NULL;
    int text_length = 0;
    double start_time, end_time;

    if (rank == 0) {
        // Read the input file
        FILE *file = fopen("input_1L.txt", "r");
        if (file == NULL) {
            perror("Could not open input.txt");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        fseek(file, 0, SEEK_END);
        long file_size = ftell(file);
        fseek(file, 0, SEEK_SET);

        text = malloc(file_size + 1);
        if (text == NULL) {
            perror("Memory allocation failed");
            fclose(file);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        fread(text, sizeof(char), file_size, file);
        fclose(file);
        text[file_size] = '\0';
        text_length = file_size;
    }

    // Broadcast the text length to all processes
    MPI_Bcast(&text_length, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Calculate local text lengths and offsets
    int local_text_length = text_length / size;
    int remainder = text_length % size;

    // Handle leftover characters in the last chunk
    if (rank == size - 1) {
        local_text_length += remainder;
    }

    // Allocate memory for the local text, including extra space for pattern length
    char *local_text = malloc(local_text_length + MAX_PATTERN_LENGTH);
    
    // Prepare for scattering
    int *sendcounts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));

    for (int i = 0; i < size; i++) {
        sendcounts[i] = (text_length / size) + (i == size - 1 ? remainder : 0);
        displs[i] = i * (text_length / size);
    }

    // Scatter the text among processes
    MPI_Scatterv(text, sendcounts, displs, MPI_CHAR, local_text, sendcounts[rank], MPI_CHAR, 0, MPI_COMM_WORLD);

    // Synchronize all processes before starting the search
    MPI_Barrier(MPI_COMM_WORLD);

  

    // Local matches storage
    int local_matches[MAX_MATCHES];
    int match_count = 0;

      // Start time measurement
    start_time = MPI_Wtime();

    // Perform the search in the local chunk of text
    bitap_search(local_text, local_text_length, pattern, rank, displs[rank], local_matches, &match_count);

     // End time measurement
    end_time = MPI_Wtime();

    // Gather all matches from all processes to the master process
    int total_matches = 0;
    MPI_Reduce(&match_count, &total_matches, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Allocate memory for all matches on the master process
    int *all_matches = NULL;
    int *all_processes = NULL;
    if (rank == 0) {
        all_matches = malloc(total_matches * sizeof(int));
        all_processes = malloc(total_matches * sizeof(int)); // Array to store process IDs
    }

    // Create an array to hold the counts of matches from each process
    int *recvcounts = NULL;
    if (rank == 0) {
        recvcounts = malloc(size * sizeof(int));
    }

    // Gather the number of matches from each process
    MPI_Gather(&match_count, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Now gather all local matches to the master process using MPI_Gatherv
    if (rank == 0) {
        // Calculate displacements for gathering
        int *displs_gather = malloc(size * sizeof(int));
        displs_gather[0] = 0;
        for (int i = 1; i < size; i++) {
            displs_gather[i] = displs_gather[i - 1] + recvcounts[i - 1];
        }

        // Now gather matches using Gatherv
        MPI_Gatherv(local_matches, match_count, MPI_INT, all_matches, recvcounts, displs_gather, MPI_INT, 0, MPI_COMM_WORLD);
        
        // Store process IDs for the matches
        int index = 0;
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < recvcounts[i]; j++) {
                all_processes[index++] = i; // Store the process ID based on the global matches gathered
            }
        }

        free(displs_gather);
        free(recvcounts);
    } else {
        // Use MPI_Gatherv to gather local matches
        MPI_Gatherv(local_matches, match_count, MPI_INT, NULL, NULL, NULL, MPI_INT, 0, MPI_COMM_WORLD);
    }

   

    // Write results to a single output file from the master process
    if (rank == 0) {
        FILE *output_file = fopen("output_1L.txt", "w");
        if (output_file == NULL) {
            perror("Error opening output file");
            free(all_matches);
            free(all_processes);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        for (int i = 0; i < total_matches; i++) {
            fprintf(output_file, "Process %d: Pattern found at global position: %d\n", all_processes[i], all_matches[i]);
        }

        printf("\nTime taken: %f seconds\n", end_time - start_time);
        fclose(output_file);
        free(all_matches);
        free(all_processes);
    }

    // Clean up memory
    free(local_text);
    free(sendcounts);
    free(displs);
    if (rank == 0) {
        free(text);
    }
    
    MPI_Finalize();
    return 0;
}
