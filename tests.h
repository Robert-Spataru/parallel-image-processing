#ifndef TESTS_H
#define TESTS_H

#include <stdint.h>
#include "utils.h"

uint8_t **generate_random_img(int num_rows, int num_cols, int rgb_min[NUM_CHANNELS], int rgb_range[NUM_CHANNELS]);

void test_phase1(const int test_resolutions[NUM_TESTCASES][2], uint8_t **const test_imgs[NUM_TESTCASES], int testcase);
void test_phase2(const int test_resolutions[NUM_TESTCASES][2], uint8_t **const test_imgs[NUM_TESTCASES], int testcase);
void test_phase3(const int test_resolutions[NUM_TESTCASES][2], uint8_t **const test_imgs[NUM_TESTCASES], int testcase);

#endif