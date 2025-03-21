/*
 *  Created by Robert Spataru
 *  
 */

#include <stdlib.h>
#include <omp.h>

#include "utils.h"
#include "parallel.h"

/*
 *  PHASE 1: compute the mean pixel value
 *  This code is buggy! Find the bug and speed it up.
 */
void mean_pixel_parallel(const uint8_t img[][NUM_CHANNELS], int num_rows, int num_cols, double mean[NUM_CHANNELS])
{
    int row, col;
    long count = (long)num_rows * num_cols;
    float reciprocal_count = ((float)1.0) / count;

    // Initialize mean array to zero 
    mean[0] = 0.0;
    mean[1]= 0.0;
    mean[2] = 0.0;


    double mean_1 = 0.0;
    double mean_2 = 0.0;
    double mean_3 = 0.0;
    int pixel_id = 0;
    // Parallel region for accumulation
    #pragma omp parallel
        {
            // Distribute the nested loops across threads
            #pragma omp for collapse(2) reduction(+: mean_1, mean_2, mean_3) private(row, col, pixel_id)
                    for (row = 0; row < num_rows; row++)
                    {
                        for (col = 0; col < num_cols; col++)
                        {
                            pixel_id = row * num_cols + col;
                            mean_1 += img[pixel_id][0];
                            mean_2 += img[pixel_id][1];
                            mean_3 += img[pixel_id][2];
                            }
                        }
                    }

    // Combine each thread's local mean sums into the shared mean array
    #pragma omp critical
            {
                mean[0] += (mean_1 * reciprocal_count);
                mean[1] += (mean_2 * reciprocal_count);
                mean[2] += (mean_3 * reciprocal_count);
            }

        }


/*
 *  PHASE 2: convert image to grayscale and record the max grayscale value along with the number of times it appears
 *  This code is NOT buggy, just sequential. Speed it up.
 */
        void grayscale_parallel(const uint8_t img[][NUM_CHANNELS], int num_rows, int num_cols,
                                uint32_t grayscale_img[][NUM_CHANNELS], uint8_t *max_gray, uint32_t *max_count)
        {
            *max_gray = 0;
            *max_count = 0;

            // Thread-local variables
            uint8_t local_max_gray = 0;
            uint32_t local_max_count = 0;
            uint32_t sum = 0;
            uint8_t gray = 0;

            #pragma omp parallel firstprivate(local_max_gray, local_max_count, sum, gray)
            {
                #pragma omp for collapse(2) nowait
                for (int row = 0; row < num_rows; row++)
                {
                    int row_start = row * num_cols;
                    for (int col = 0; col < num_cols; col++)
                    {
                        int pixel_id = row_start + col;
                        sum = img[pixel_id][0] + img[pixel_id][1] + img[pixel_id][2];
                        gray = (uint32_t)(sum / NUM_CHANNELS);

                        grayscale_img[pixel_id][0] = gray;
                        grayscale_img[pixel_id][1] = gray;
                        grayscale_img[pixel_id][2] = gray;

                        if (gray > local_max_gray)
                        {
                            local_max_gray = gray;
                            local_max_count = NUM_CHANNELS;
                        }
                        else if (gray == local_max_gray)
                        {
                            local_max_count += NUM_CHANNELS;
                        }
                    }
                }

                // Aggregate thread-local results into global variables
                #pragma omp critical
                {
                    if (local_max_gray > *max_gray)
                    {
                        *max_gray = local_max_gray;
                        *max_count = local_max_count;
                    }
                    else if (local_max_gray == *max_gray)
                    {
                        *max_count += local_max_count;
                    }
                }
            }
        }

/*
 *  PHASE 3: perform convolution on image
 *  This code is NOT buggy, just sequential. Speed it up.
 */
void convolution_parallel(const uint8_t padded_img[][NUM_CHANNELS], int num_rows, int num_cols, const uint32_t kernel[], int kernel_size, uint32_t convolved_img[][NUM_CHANNELS])
{
    int kernel_norm, i;
    int conv_rows, conv_cols;
    uint32_t r_sum, g_sum, b_sum; // substitutes having to go into memory with convolved_img[row * conv_cols + col][ch]
    const int TILE_WIDTH = 32;
    const int TILE_HEIGHT = 32;

    // compute kernel normalization factor
    kernel_norm = 0;
    #pragma omp parallel for reduction(+:kernel_norm)
    for (i = 0; i < kernel_size * kernel_size; i++)
    {
        kernel_norm += kernel[i];
    }
    float kernel_norm_reciprocal = ((float)1.0)/ kernel_norm;

    // compute dimensions of convolved image
    conv_rows = num_rows - kernel_size + 1;
    conv_cols = num_cols - kernel_size + 1;

    // perform convolution
    #pragma omp parallel 
    {
        #pragma omp for collapse(2) private(r_sum, g_sum, b_sum)
        for (int row_tile = 0; row_tile < conv_rows; row_tile += TILE_HEIGHT)
        {
            for (int col_tile = 0; col_tile < conv_cols; col_tile += TILE_WIDTH)
            {
                int row_tile_end = (row_tile + TILE_HEIGHT) < conv_rows ? row_tile + TILE_HEIGHT : conv_rows;
                int col_tile_end = (col_tile + TILE_WIDTH) < conv_cols ? col_tile + TILE_WIDTH : conv_cols;
        
                for (int row = row_tile; row < row_tile_end; row++)
                {
                    for (int col = col_tile; col < col_tile_end; col++)
                    {
                        int pixel_id = row * conv_cols + col;
                        r_sum = 0;
                        g_sum = 0;
                        b_sum = 0;
                        
                        for (int kernel_row = 0; kernel_row < kernel_size; kernel_row++)
                        {
                            for (int kernel_col = 0; kernel_col < kernel_size; kernel_col++)
                            {
                                int kernel_id = kernel_row * kernel_size + kernel_col;
                                int padded_index = (row + kernel_row) * num_cols + (col + kernel_col);
                                r_sum += padded_img[padded_index][0] * kernel[kernel_id];
                                g_sum += padded_img[padded_index][1] * kernel[kernel_id];
                                b_sum += padded_img[padded_index][2] * kernel[kernel_id];
                            }
                        }
                        convolved_img[pixel_id][0] = (uint32_t)(r_sum * kernel_norm_reciprocal);
                        convolved_img[pixel_id][1] = (uint32_t)(g_sum * kernel_norm_reciprocal);
                        convolved_img[pixel_id][2] = (uint32_t)(b_sum * kernel_norm_reciprocal);
                    }
                }
            }
        }
    } 
}
