#pragma once

#include "darknet.h"

#include <sys/time.h>
#include <assert.h>

#include "../src/parser.h"


float** calculate_confusion_matrix (char*, char*, char*, char*);

float safe_division (float, float);

float calculate_accuracy (float**, int);

float calculate_precision (float**, int);

float calculate_recall (float**, int);

float calculate_neg_precision (float**, int);

float calculate_specificity (float**, int);

void output_training_log (FILE*, int, float**, float**, int, size_t);




typedef float (*EvaluationMetric)(float**, int);


typedef enum { ACCURACY, PRECISION, RECALL, NEG_PRECISION, SPECIFICITY, NUM_METRICS } TRAIN_METRIC;

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


extern const EvaluationMetric evaluation_metrics[NUM_METRICS];
extern const char* evaluation_metric_names[NUM_METRICS];