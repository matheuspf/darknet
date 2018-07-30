#pragma once

#include "darknet.h"

#include <sys/time.h>
#include <assert.h>

#include "../src/parser.h"


float** calculate_confusion_matrix (char*, char*, char*, char*);

float calculate_metric (float, float);

float calculate_accuracy (float**, int);

float calculate_precision (float**, int);


typedef float (*Evaluation_Metric)(float**, int);


typedef enum { ACCURACY, PRECISION, F1_SCORE, LOSS, NUM_METRICS } TRAIN_METRIC;

typedef struct
{
    TRAIN_METRIC metric;
    int eval_epochs;
    int max_epochs;
    int patience;
    int seed;
    char* training_log;

} SSM_Params;


extern const Evaluation_Metric evaluation_metrics[NUM_METRICS];