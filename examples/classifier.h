#pragma once

#include "darknet.h"

#include <sys/time.h>
#include <assert.h>

#include "../src/parser.h"
#include "../src/metrics.h"


typedef enum { ACCURACY, PRECISION, RECALL, NPV, SPECIFICITY, F1_SCORE, NUM_METRICS } TRAIN_METRIC;

typedef struct
{
    TRAIN_METRIC metric;
    int eval_epochs;
    int max_epochs;
    int patience;
    int seed;
    char* log_file;
    size_t log_output;

} SSM_Params;


extern const ScoreMetric evaluation_metrics[NUM_METRICS];
extern const char* evaluation_metric_names[NUM_METRICS];


void train_classifier_valid(char*, char*, char*, int*, int, int, SSM_Params);

float** get_predictions (char*, char*, char*, char*, float**, int*);

void output_training_log (FILE*, int, int*, float**, int, int*, float**, int, int, size_t);