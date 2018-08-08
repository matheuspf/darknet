#pragma once

#include "darknet.h"

#include <sys/time.h>
#include <sys/stat.h>
#include <assert.h>

#include "../src/utils.h"
#include "../src/parser.h"
#include "../src/metrics.h"


typedef enum { ACCURACY, PRECISION, RECALL, NPV, SPECIFICITY, F1_SCORE, NUM_METRICS } TRAIN_METRIC;

typedef struct
{
    int* y_true_train;
    float** y_score_train;
    int* y_true_valid;
    float** y_score_valid;
    int N_train;
    int N_valid;
    int classes;

} EpochResults;

typedef struct
{
    TRAIN_METRIC metric;
    int eval_epochs;
    int max_epochs;
    int patience;
    int seed;
    char* log_file;
    size_t log_output;
    char* hyper_param_file;
    // The maximum number of predictions that will be made to calculate
    // metrics after the end of each epoch.
    // Predictions are counted separately for training and validation datasets.
    int max_predictions;
    int verbose;

} SSM_Params;


extern const ScoreMetric evaluation_metrics[NUM_METRICS];
extern const char* evaluation_metric_names[NUM_METRICS];


EpochResults* new_epoch_results(int, int, int);

void destroy_epoch_results(EpochResults*);

void copy_epoch_results(EpochResults*, int*, float**, int*, float**);

float *get_regression_values(char**, int);

void write_summary_hyperparameters(FILE*, const network*);

void write_summary_metrics(FILE*, EpochResults*);

void save_training_summary(network*, FILE*, EpochResults*);


void write_net_file (network*, char*);

void train_classifier_valid(char*, char*, char*, int*, int, int, SSM_Params);

void get_predictions (network*, char**, char**, int, int, float**, int*, int);

void output_training_log (FILE*, int, float, double, const EpochResults, size_t);

char* full_time_stamp ();

void copy_file_str (FILE*, FILE*);