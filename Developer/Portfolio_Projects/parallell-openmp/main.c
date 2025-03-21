#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

#include "tests.h"
#include "utils.h"

// Display usage info
static void usage(char *cmd)
{
    printf("Usage: %s [-h] [-p <n>] [-r <n>] [-c <n>]\n", cmd);
    printf("  -h        Print this message.\n");
    printf("  -p <n>    Test only phase n.\n");
    exit(1);
}

int main(int argc, char *argv[])
{
    char c;
    int phase;

    int num_rows, num_cols, log_row_range, log_col_range, testcase, counter, ch;
    int rgb_val1, rgb_val2, rgb_min[NUM_CHANNELS], rgb_range[NUM_CHANNELS];

    int test_resolutions[NUM_TESTCASES][2];
    uint8_t **test_imgs[NUM_TESTCASES];

    /* parse command line args */
    phase = 0, testcase = 0;
    while ((c = getopt(argc, argv, "hp:t:")) != -1)
    {
        switch (c)
        {
        case 'h': /* help */
            usage(argv[0]);
            break;
        case 'p': /* select phase to test */
            phase = atoi(optarg);
            if (phase < 1 || phase > 3)
                usage(argv[0]);
            break;
        default:
            usage(argv[0]);
        }
    }

    /* randomly generate an image to test */
    srand(time(NULL));
    for (ch = 0; ch < NUM_CHANNELS; ch++)
    {
        rgb_range[ch] = -1;
        while (rgb_range[ch] < MIN_VALUE_RANGE)
        {
            rgb_val1 = rand() % VALUE_RANGE;
            rgb_val2 = rand() % VALUE_RANGE;
            rgb_range[ch] = 1 + abs(rgb_val2 - rgb_val1);
        }
        rgb_min[ch] = MIN(rgb_val1, rgb_val2);
    }

    counter = 0;
    for (log_row_range = MIN_LOG_DIM; log_row_range <= MAX_LOG_DIM; log_row_range++)
    {
        for (log_col_range = log_row_range; log_col_range <= MAX_LOG_DIM; log_col_range++)
        {
            num_rows = 1 << log_row_range;
            num_cols = 1 << log_col_range;
            test_resolutions[counter][0] = num_rows;
            test_resolutions[counter][1] = num_cols;
            test_imgs[counter] = generate_random_img(num_rows, num_cols, rgb_min, rgb_range);
            counter++;
        }
    }

    printf("Use CTRL-C to exit if necessary.\n");
    printf("Make sure to take care of possible data races!\n\n");

    /* testing */
    switch (phase)
    {
    case 1:
        test_phase1(test_resolutions, test_imgs, testcase);
        break;
    case 2:
        test_phase2(test_resolutions, test_imgs, testcase);
        break;
    case 3:
        test_phase3(test_resolutions, test_imgs, testcase);
        break;
    default:
        test_phase1(test_resolutions, test_imgs, testcase);
        test_phase2(test_resolutions, test_imgs, testcase);
        test_phase3(test_resolutions, test_imgs, testcase);
    }

    /* cleanup */
    for (testcase = 0; testcase < NUM_TESTCASES; testcase++)
        free(test_imgs[testcase]);

    return 0;
}
