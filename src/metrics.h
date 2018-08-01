#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>


typedef float (*ScoreMetric) (int*, float**, int, int);

typedef float (*PredictionMetric) (int*, int*, int, int);

typedef float (*ConfusionMatrixMetric) (float**, int);

typedef float** (*CurveMetric) (int*, float*, int);


int pair_float_cmp (const void*, const void*);

void paired_sort (float*, float*, int);

float safe_division (float, float);

float naive_auc (float*, float*, int);


void* new_mat (int, int, size_t);

void copy_mat(void** dst, void** src, int rows, int cols, size_t size);

void del_mat (void*, int);


int* top_prediction (float**, int, int);

float** confusion_matrix (int*, int*, int, int);


float prediction_metric (int*, int*, int, int, ConfusionMatrixMetric);

float confusion_matrix_metric (int*, float**, int, int, ConfusionMatrixMetric);

float* fixed_metric_score (int*, float*, int, float, CurveMetric);


float accuracy_score (int*, float**, int, int);

float precision_score (int*, float**, int, int);

float recall_score (int*, float**, int, int);

float npv_score (int*, float**, int, int);

float specificity_score (int*, float**, int, int);

float f1_score (int*, float**, int, int);


float accuracy_score_pred (int*, int*, int, int);

float precision_score_pred (int*, int*, int, int);

float recall_score_pred (int*, int*, int, int);

float npv_score_pred (int*, int*, int, int);

float specificity_score_pred (int*, int*, int, int);

float f1_score_pred (int*, int*, int, int);


float accuracy_score_conf (float**, int);

float precision_score_conf (float**, int);

float recall_score_conf (float**, int);

float npv_score_conf (float**, int);

float specificity_score_conf (float**, int);

float f_score_conf (float**, int, float);

float f1_score_conf (float**, int);


float** precision_recall_curve (int*, float*, int);

float** roc_curve (int*, float*, int);

float** npv_tnr_curve (int*, float*, int);


float* fixed_recall_score (int*, float*, int, float);

float* fixed_specificity_score (int*, float*, int, float);
