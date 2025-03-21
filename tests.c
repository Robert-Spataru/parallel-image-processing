#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "sequential.h"
#include "parallel.h"
#include "utils.h"

#include "tests.h"

static const uint32_t gblur_3x3_kernel[] = {
    1, 2, 1,
    2, 4, 2,
    1, 2, 1};

static const uint32_t gblur_5x5_kernel[] = {
    1, 4, 6, 4, 1,
    4, 16, 24, 16, 4,
    6, 24, 36, 24, 6,
    4, 16, 24, 16, 4,
    1, 4, 6, 4, 1};

static inline void print_time(double seqDelay, double parDelay)
{
    printf("Sequential version: %lf sec, Parallel version: %lf sec\n", seqDelay, parDelay);
}

static inline void print_mean_pixel(const double arr[])
{
    int c;
    printf("[");
    for (c = 0; c < NUM_CHANNELS - 1; c++)
    {
        printf("%f, ", arr[c]);
    }
    printf("%f]", arr[NUM_CHANNELS - 1]);
}

static inline void print_image(const uint8_t img[][NUM_CHANNELS], int num_rows, int num_cols)
{
    int row, col, ch, i;
    for (row = 0; row < num_rows; row++)
    {
        for (col = 0; col < num_cols; col++)
        {
            i = row * num_cols + col;
            printf("(");
            for (ch = 0; ch < NUM_CHANNELS - 1; ch++)
            {
                printf("%d,", img[i][ch]);
            }
            printf("%d)\t", img[i][NUM_CHANNELS - 1]);
        }
        printf("\n");
    }
}

static inline void print_output_image(const uint32_t img[][NUM_CHANNELS], int num_rows, int num_cols)
{
    int row, col, ch, i;
    for (row = 0; row < num_rows; row++)
    {
        for (col = 0; col < num_cols; col++)
        {
            i = row * num_cols + col;
            printf("(");
            for (ch = 0; ch < NUM_CHANNELS - 1; ch++)
            {
                printf("%d,", img[i][ch]);
            }
            printf("%d)\t", img[i][NUM_CHANNELS - 1]);
        }
        printf("\n");
    }
}

static inline int compare_images(const uint32_t sequential_img[][NUM_CHANNELS], const uint32_t parallel_img[][NUM_CHANNELS], int size)
{
    int i, c;
    for (i = 0; i < size; i++)
    {
        for (c = 0; c < NUM_CHANNELS; c++)
        {
            if (sequential_img[i][c] != parallel_img[i][c])
            {
                return 1;
            }
        }
    }
    return 0;
}

uint8_t **generate_random_img(int num_rows, int num_cols, int rgb_min[NUM_CHANNELS], int rgb_range[NUM_CHANNELS])
{
    assert(rgb_min != NULL);
    assert(rgb_range != NULL);

    int row, col, ch, rgb_val;

    uint8_t(*img)[NUM_CHANNELS] = malloc(sizeof(uint8_t) * num_rows * num_cols * NUM_CHANNELS);
    for (row = 0; row < num_rows; row++)
    {
        for (col = 0; col < num_cols; col++)
        {
            for (ch = 0; ch < NUM_CHANNELS; ch++)
            {
                rgb_val = rgb_min[ch] + (rand() % rgb_range[ch]);
                img[row * num_cols + col][ch] = rgb_val;
            }
        }
    }

    return (uint8_t **)img;
}

/* PHASE 1 TESTING */
double test_mean_pixel(const uint8_t img[][NUM_CHANNELS], int num_rows, int num_cols)
{
    int ch;
    double seqMean[NUM_CHANNELS], parMean[NUM_CHANNELS];

    struct timespec start, finish;
    double seqDelay, parDelay;
    double speedup;

    printf("Running phase 1 on a %dx%d image...\n", num_rows, num_cols);

    clock_gettime(CLOCK_MONOTONIC, &start);
    mean_pixel_seq(img, num_rows, num_cols, seqMean);
    clock_gettime(CLOCK_MONOTONIC, &finish);

    seqDelay = (finish.tv_sec - start.tv_sec);
    seqDelay += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

    clock_gettime(CLOCK_MONOTONIC, &start);
    mean_pixel_parallel(img, num_rows, num_cols, parMean);
    clock_gettime(CLOCK_MONOTONIC, &finish);

    parDelay = (finish.tv_sec - start.tv_sec);
    parDelay += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    print_time(seqDelay, parDelay);

    for (ch = 0; ch < NUM_CHANNELS; ch++)
    {
        if (fabs(seqMean[ch] - parMean[ch]) > EPS || isnan(parMean[ch]))
        {
            printf("Your phase 1 results are incorrect.\n");
            printf("The expected result was ");
            print_mean_pixel(seqMean);
            printf(".\nThe observed result was ");
            print_mean_pixel(parMean);
            printf(".\n\n");
            return 0.0;
        }
    }

    speedup = seqDelay / parDelay;
    printf("\n");
    return speedup;
}

void test_phase1(const int test_resolutions[NUM_TESTCASES][2], uint8_t **const test_imgs[NUM_TESTCASES], int testcase)
{
    assert(0 <= testcase && testcase <= NUM_TESTCASES);

    printf("--------------TEST PHASE 1---------------\n");
    int num_rows, num_cols;
    double speedup[NUM_TESTCASES];
    uint8_t(*test_img)[NUM_CHANNELS];

    if (testcase == 0)
    {
        // test all resolution settings
        for (testcase = 0; testcase < NUM_TESTCASES; testcase++)
        {
            num_rows = test_resolutions[testcase][0];
            num_cols = test_resolutions[testcase][1];
            test_img = (uint8_t(*)[NUM_CHANNELS])test_imgs[testcase];
            speedup[testcase] = test_mean_pixel(test_img, num_rows, num_cols);
        }

        printf("SUMMARY:\n");
        for (testcase = 0; testcase < NUM_TESTCASES; testcase++)
        {
            num_rows = test_resolutions[testcase][0];
            num_cols = test_resolutions[testcase][1];
            printf("[PHASE 1][%dx%d] Speedup: %lf\n", num_rows, num_cols, speedup[testcase]);
        }
    }
    else
    {
        // test selected resolution setting
        num_rows = test_resolutions[testcase - 1][0];
        num_cols = test_resolutions[testcase - 1][1];
        test_img = (uint8_t(*)[NUM_CHANNELS])test_imgs[testcase - 1];
        speedup[testcase - 1] = test_mean_pixel(test_img, num_rows, num_cols);
        printf("SUMMARY:\n");
        printf("[PHASE 1][%dx%d] Speedup: %lf\n", num_rows, num_cols, speedup[testcase - 1]);
    }
}

/* PHASE 2 TESTING */
double test_greyscale(const uint8_t img[][NUM_CHANNELS], int num_rows, int num_cols)
{
    int error = 0;
    uint32_t(*sequential_grayscale_img)[NUM_CHANNELS], (*parallel_grayscale_img)[NUM_CHANNELS];
    uint8_t sequential_max_gray, parallel_max_gray;
    uint32_t sequential_max_count, parallel_max_count;

    double speedup;
    struct timespec start, finish;
    double seqDelay, parDelay;

    // dynamically allocate grayscale images
    sequential_grayscale_img = malloc(sizeof(uint32_t) * num_rows * num_cols * NUM_CHANNELS);
    parallel_grayscale_img = malloc(sizeof(uint32_t) * num_rows * num_cols * NUM_CHANNELS);

    printf("Running phase 2 on a %dx%d image...\n", num_rows, num_cols);

    clock_gettime(CLOCK_MONOTONIC, &start);
    grayscale_seq(img, num_rows, num_cols, sequential_grayscale_img, &sequential_max_gray, &sequential_max_count);
    clock_gettime(CLOCK_MONOTONIC, &finish);

    seqDelay = (finish.tv_sec - start.tv_sec);
    seqDelay += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

    clock_gettime(CLOCK_MONOTONIC, &start);
    grayscale_parallel(img, num_rows, num_cols, parallel_grayscale_img, &parallel_max_gray, &parallel_max_count);
    clock_gettime(CLOCK_MONOTONIC, &finish);

    parDelay = (finish.tv_sec - start.tv_sec);
    parDelay += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    print_time(seqDelay, parDelay);

    error = compare_images(sequential_grayscale_img, parallel_grayscale_img, num_rows * num_cols);
    if (sequential_max_count != parallel_max_count || sequential_max_gray != parallel_max_gray)
    {
        error += 2;
    }

    if (error % 2 == 1)
    {
        printf("Your phase 2 image results are incorrect.\n\n");
    }

    if (error >= 2)
    {
        printf("Your phase 2 max gray results are incorrect.\n");
        printf("The expected max gray value was %d, appearing %d times.\n", sequential_max_gray, sequential_max_count);
        printf("The observed max gray value was %d, appearing %d times.\n\n", parallel_max_gray, parallel_max_count);
    }

    free(sequential_grayscale_img);
    free(parallel_grayscale_img);

    if (error != 0)
    {
        return 0.0;
    }

    speedup = seqDelay / parDelay;
    printf("\n");
    return speedup;
}

void test_phase2(const int test_resolutions[NUM_TESTCASES][2], uint8_t **const test_imgs[NUM_TESTCASES], int testcase)
{
    assert(0 <= testcase && testcase <= NUM_TESTCASES);

    printf("--------------TEST PHASE 2---------------\n");
    int num_rows, num_cols;
    double speedup[NUM_TESTCASES];
    uint8_t(*test_img)[NUM_CHANNELS];
    if (testcase == 0)
    {
        // test all resolution settings
        for (testcase = 0; testcase < NUM_TESTCASES; testcase++)
        {
            num_rows = test_resolutions[testcase][0];
            num_cols = test_resolutions[testcase][1];
            test_img = (uint8_t(*)[NUM_CHANNELS])test_imgs[testcase];
            speedup[testcase] = test_greyscale(test_img, num_rows, num_cols);
        }

        printf("SUMMARY:\n");
        for (testcase = 0; testcase < NUM_TESTCASES; testcase++)
        {
            num_rows = test_resolutions[testcase][0];
            num_cols = test_resolutions[testcase][1];
            printf("[PHASE 2][%dx%d] Speedup: %lf\n", num_rows, num_cols, speedup[testcase]);
        }
    }
    else
    {
        // test selected resolution setting
        num_rows = test_resolutions[testcase - 1][0];
        num_cols = test_resolutions[testcase - 1][1];
        test_img = (uint8_t(*)[NUM_CHANNELS])test_imgs[testcase - 1];
        speedup[testcase - 1] = test_greyscale(test_img, num_rows, num_cols);
        printf("SUMMARY:\n");
        printf("[PHASE 2][%dx%d] Speedup: %lf\n", num_rows, num_cols, speedup[testcase - 1]);
    }
}

/* PHASE 3 TESTING */
double test_convolution(const uint8_t img[][NUM_CHANNELS], int num_rows, int num_cols, const uint32_t kernel[], int kernel_size)
{
    int error = 0;
    int padding, pad_rows, pad_cols;
    int row, col, ch;

    uint8_t(*padded_img)[NUM_CHANNELS];
    uint32_t(*sequential_conv_img)[NUM_CHANNELS], (*parallel_conv_img)[NUM_CHANNELS];

    double speedup;
    struct timespec start, finish;
    double seqDelay, parDelay;

    assert(kernel_size % 2 == 1);
    padding = (kernel_size - 1) / 2;
    pad_rows = num_rows + 2 * padding;
    pad_cols = num_cols + 2 * padding;

    // dynamically allocate convolved and padded images
    sequential_conv_img = malloc(sizeof(uint32_t) * num_rows * num_cols * NUM_CHANNELS);
    parallel_conv_img = malloc(sizeof(uint32_t) * num_rows * num_cols * NUM_CHANNELS);

    padded_img = malloc(sizeof(uint8_t) * pad_rows * pad_cols * NUM_CHANNELS);
    for (row = 0; row < num_rows; row++)
    {
        for (col = 0; col < num_cols; col++)
        {
            for (ch = 0; ch < NUM_CHANNELS; ch++)
            {
                padded_img[(row + padding) * pad_cols + col + padding][ch] = img[row * num_cols + col][ch];
            }
        }
    }

    printf("Running phase 3 on a %dx%d image with a %dx%d kernel...\n", num_rows, num_cols, kernel_size, kernel_size);

    clock_gettime(CLOCK_MONOTONIC, &start);
    convolution_seq(padded_img, pad_rows, pad_cols, kernel, kernel_size, sequential_conv_img);
    clock_gettime(CLOCK_MONOTONIC, &finish);

    seqDelay = (finish.tv_sec - start.tv_sec);
    seqDelay += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

    clock_gettime(CLOCK_MONOTONIC, &start);
    convolution_parallel(padded_img, pad_rows, pad_cols, kernel, kernel_size, parallel_conv_img);
    clock_gettime(CLOCK_MONOTONIC, &finish);

    parDelay = (finish.tv_sec - start.tv_sec);
    parDelay += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    print_time(seqDelay, parDelay);

    error = compare_images(sequential_conv_img, parallel_conv_img, num_rows * num_cols);

    free(sequential_conv_img);
    free(parallel_conv_img);
    free(padded_img);

    if (error != 0)
    {
        printf("Your phase 3 image results are incorrect.\n\n");
        return 0.0;
    }

    speedup = seqDelay / parDelay;
    printf("\n");
    return speedup;
}

void test_phase3(const int test_resolutions[NUM_TESTCASES][2], uint8_t **const test_imgs[NUM_TESTCASES], int testcase)
{
    assert(0 <= testcase && testcase <= NUM_TESTCASES);

    printf("--------------TEST PHASE 3---------------\n");
    int num_rows, num_cols;
    double speedup_3x3[NUM_TESTCASES], speedup_5x5[NUM_TESTCASES];
    uint8_t(*test_img)[NUM_CHANNELS];
    if (testcase == 0)
    {
        // test all resolution settings
        for (testcase = 0; testcase < NUM_TESTCASES; testcase++)
        {
            num_rows = test_resolutions[testcase][0];
            num_cols = test_resolutions[testcase][1];
            test_img = (uint8_t(*)[NUM_CHANNELS])test_imgs[testcase];
            speedup_3x3[testcase] = test_convolution(test_img, num_rows, num_cols, gblur_3x3_kernel, 3);
            speedup_5x5[testcase] = test_convolution(test_img, num_rows, num_cols, gblur_5x5_kernel, 5);
        }

        printf("SUMMARY:\n");
        for (testcase = 0; testcase < NUM_TESTCASES; testcase++)
        {
            num_rows = test_resolutions[testcase][0];
            num_cols = test_resolutions[testcase][1];
            printf("[PHASE 3][%dx%d][3x3 KERNEL] Speedup: %lf\n", num_rows, num_cols, speedup_3x3[testcase]);
            printf("[PHASE 3][%dx%d][5x5 KERNEL] Speedup: %lf\n", num_rows, num_cols, speedup_5x5[testcase]);
        }
    }
    else
    {
        // test selected resolution setting
        num_rows = test_resolutions[testcase - 1][0];
        num_cols = test_resolutions[testcase - 1][1];
        test_img = (uint8_t(*)[NUM_CHANNELS])test_imgs[testcase - 1];
        speedup_3x3[testcase - 1] = test_convolution(test_img, num_rows, num_cols, gblur_3x3_kernel, 3);
        speedup_5x5[testcase - 1] = test_convolution(test_img, num_rows, num_cols, gblur_5x5_kernel, 5);
        printf("SUMMARY:\n");
        printf("[PHASE 3][%dx%d][3x3 KERNEL] Speedup: %lf\n", num_rows, num_cols, speedup_3x3[testcase - 1]);
        printf("[PHASE 3][%dx%d][5x5 KERNEL] Speedup: %lf\n", num_rows, num_cols, speedup_5x5[testcase - 1]);
    }
}