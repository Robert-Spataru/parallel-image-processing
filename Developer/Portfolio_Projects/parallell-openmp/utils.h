#ifndef UTILS_H
#define UTILS_H

// parameters for generating random images
// #define MIN_LOG_DIM 11
// #define MAX_LOG_DIM 12
#define MIN_VALUE_RANGE 32

/* DO NOT MODIFY BELOW THIS LINE */

// #define MIN_VALUE_RANGE 32
#define MIN_LOG_DIM 11
#define MAX_LOG_DIM 12
#define NUM_TESTCASES ((MAX_LOG_DIM - MIN_LOG_DIM + 2) * (MAX_LOG_DIM - MIN_LOG_DIM + 1) / 2)

// number of RGB image channels
#define NUM_CHANNELS 3

// range of pixel values
#define VALUE_RANGE 256

// tolerance for double comparison
#define EPS 1e-12

// min and max helpers
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

#endif