#include "metrics.h"


/** @brief Compare two pairs of floats
 *  @details If a[0] == b[0], compare a[1] with b[1]
 *  @param a First const pointer to a pair of floats
 *  @param b Second const pointer to a pair of floats
 *  @return -1 if a < b
 *           1 if a > b
 *           0 if a == b
**/
int pair_float_cmp (const void* a, const void* b)
{
    float a0 = ((float*)a)[0], a1 = ((float*)a)[1];
    float b0 = ((float*)b)[0], b1 = ((float*)b)[1];

    return a0 < b0 ? -1 : (a0 > b0 ? 1 : a1 - b1);
}

/** @brief Sort the vectors together, as if there was a single vector of pairs
 *  @details Given two float vectors, the function sorts them together, assuring that the relative
 *           ordering between the vectors stays the same. The first vector has precedence in the 
 *           sorting procedure. The original vectors are modified.
 * 
 *  @param v First vector of floats, having precedence in the sorting
 *  @param a Second vector of floats
 *  @param n Size of the vectors
 *  @return -1 if a < b
 *           1 if a > b
 *           0 if a == b
**/
void paired_sort (float* v, float* u, int n)
{
    int i;
    float* pair = malloc(2*n*sizeof(float));

    for(i = 0; i < n; ++i)
        pair[2*i] = v[i], pair[2*i+1] = u[i];

    qsort(pair, n, 2*sizeof(float), pair_float_cmp);

    for(i = 0; i < n; ++i)
        v[i] = pair[2*i], u[i] = pair[2*i+1];

    free(pair);
}

/// Divide @c num by @c den safely, returning 0 if @c den is very small
float safe_division (float num, float den)
{
    return fabs(den) < 1e-8 ? 0.0 : num / den;
}

/// Naive trapezoidal integral evaluation
float naive_auc (float* x, float* y, int n)
{
    int i;
    double auc = 0.0;

    for(i = 0; i < n-1; ++i)
        auc += fabsf(x[i+1] - x[i]) * ((y[i] + y[i+1])/2);

    return auc;
}

/** @brief Create a matrix (a pointer of pointers) given the number of rows, cols and the size of the base type
 *  @details Easy to use, not very fast (memory is not sequential). Always initialized to 0.
 *  @param rows Number of rows
 *  @param cols Number of cols
 *  @param size Size of the base element type
 *  @return A pointer to pointers of void, pointing to the newly created matrix
*/
void* new_mat (int rows, int cols, size_t size)
{
    int i;
    void** mat = malloc(rows * sizeof(void*));

    for(i = 0; i < rows; ++i)
        mat[i] = calloc(cols, size);
    
    return mat;
}

/**
 * @brief Copy the contents of the matrix <dst> into the matrix <src>
 * @param dst Destination matrix
 * @param src Source matrix
 * @param rows Number of rows
 * @param cols Number of columns
 * @param size Size of the base element type
 */
void copy_mat(void** dst, void** src, int rows, int cols, size_t size)
{
    int i;

    for(i = 0; i < rows; ++i)
        memcpy(dst[i], src[i], cols * size);
}


/** @brief Free the memory of a matrix (pointer to pointers)
 *  @param mat Matrix to be fred (cast to a void pointer to avoid warnings)
 *  @param rows Number of rows of the matrix (the number of cols is not necessary)
*/
void del_mat (void* mat, int rows)
{
    int i;

    for(i = 0; i < rows; ++i)
        free(((void**)mat)[i]);

    free(mat);
}


/** @brief Calculate the top 1 prediction given a matrix of prections
 *  @details Given a matrix @c y_pred, containing <tt> n x classes </tt> elements, for each 
 *           of the @c n observations the top 1 prediction (of the @c classes total) is saved
 * 
 *  @param y_pred A <tt> n x classes </tt> float matrix of scores (does not need to be normalized)
 *  @param n Number of observations
 *  @param classes Number of classes
 *  @return A vector of integers with size @c n, containing for each observation the top 1 prediction
*/
int* top_prediction (float** y_pred, int n, int classes)
{
    int i, j;
    int* top_pred = malloc(n * sizeof(int));

    for(i = 0; i < n; ++i)
    {
        int pos = 0;
        float best = -1.0;

        for(j = 0; j < classes; ++j) if(best < y_pred[i][j])
        {
            best = y_pred[i][j];
            pos = j;
        }

        top_pred[i] = pos;
    }

    return top_pred;
}


/** @brief Calculate a confusion matrix given the true observations @c y_true and the predictions @c y_pred
 *  @details Works for any number of classes
 *  @param y_true The vector of true observations, containing the actual corresponding class for the observation
 *  @param y_pred The vector of predictions, containing the value of the most likely class of that observation
 *  @param n Size of @c y_true
 *  @param classes Total number of classes
 *  @return Return a confusion matrix, a <tt>classes x classes</tt> float matrix containing statistics about
 *          the classification
**/
float** confusion_matrix (int* y_true, int* y_pred, int n, int classes)
{
    int i;
    float** mat = (float**)new_mat(classes, classes, sizeof(float));

    for(i = 0; i < n; ++i)
        mat[y_true[i]][y_pred[i]]++;

    return mat;
}


float** confusion_matrix_score (int* y_true, float** y_score, int n, int classes)
{
    int* y_pred = top_prediction(y_score, n, classes);

    float** conf_mat = confusion_matrix(y_true, y_pred, n, classes);

    free(y_pred);

    return conf_mat;
}


float prediction_metric (int* y_true, int* y_pred, int n, int classes, ConfusionMatrixMetric metric)
{
    float** conf_mat = confusion_matrix(y_true, y_pred, n, classes);

    float res = metric(conf_mat, classes);

    del_mat(conf_mat, classes);

    return res;
}


float confusion_matrix_metric (int* y_true, float** y_score, int n, int classes, ConfusionMatrixMetric metric)
{
    int* y_pred = top_prediction(y_score, n, classes);

    float res = prediction_metric(y_true, y_pred, n, classes, metric);

    free(y_pred);

    return res;
}



/** @brief Metrics based on the confusion matrix
 *  @details For two classes all metrics have a very clear definition. When the number of classes is greater
 *           than two, there are different methods to calculate them. The current algorithm is the 'macro'
 *           procedure, described here http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
 *  @param confusion_matrix A <tt>classes x classes</tt> float confusion matrix
 *  @param classes The total number of classes
**/
//@{

/** @brief Calculate the accuracy score given a confusion matrix
 *  @details @f$accuracy = \frac{TP + TN}{TP + FP + TN + FN}@f$
 *  @note As the accuracy is symetric, the same formula also works for <tt>classes > 2</tt> 
**/
float accuracy_score_conf (float** confusion_matrix, int classes)
{
    int i, j;
    float sum = 0.0, diag = 0.0;

    for(i = 0; i < classes; ++i)
    {
        for(j = 0; j < classes; ++j)
            sum += confusion_matrix[i][j];

        diag += confusion_matrix[i][i];
    }

    return diag / sum;
}


/** @brief Calculate the precision score given a confusion matrix
 *  @details @f$precision = \frac{TP}{TP + FP}@f$
**/
float precision_score_conf (float** confusion_matrix, int classes)
{
    if(classes == 2)
        return safe_division(confusion_matrix[1][1], confusion_matrix[1][1] + confusion_matrix[0][1]);

    int i, j;
    float mean = 0.0;

    for(i = 0; i < classes; ++i)
    {
        float sum = 0.0;

        for(j = 0; j < classes; ++j)
            sum += confusion_matrix[j][i];

        mean += safe_division(confusion_matrix[i][i], sum);
    }

    return mean / classes;
}


/** @brief Calculate the recall score given a confusion matrix
 *  @details @f$recall = \frac{TP}{TP + FN}@f$
**/
float recall_score_conf (float** confusion_matrix, int classes)
{
    if(classes == 2)
        return safe_division(confusion_matrix[1][1], confusion_matrix[1][1] + confusion_matrix[1][0]);

    int i, j;
    float mean = 0.0;

    for(i = 0; i < classes; ++i)
    {
        float sum = 0.0;

        for(j = 0; j < classes; ++j)
            sum += confusion_matrix[i][j];

        mean += (confusion_matrix[i][i] / sum);
    }

    return mean / classes;
}

/** @brief Calculate the negative predictive value (the equivalent of the precision for the 'negative' class) given a confusion matrix
 *  @details @f$precision = \frac{TN}{TN + FN}@f$
**/
float npv_score_conf (float** confusion_matrix, int classes)
{
    if(classes == 2)
        return safe_division(confusion_matrix[0][0], confusion_matrix[0][0] + confusion_matrix[1][0]);

    int i, j, k;
    float mean = 0.0;

    for(i = 0; i < classes; ++i)
    {
        float num = 0.0, den = 0.0;

        for(j = 0; j < classes; ++j)
            for(k = 0; k < classes; ++k)
                num += confusion_matrix[j][k] * (i != k), den += confusion_matrix[j][k];

        mean += (num / den);
    }

    return mean / classes;
}

/** @brief Calculate the specificity or true negative ratio score (the equivalent of the recall for the 'negative' class) given a confusion matrix
 *  @details @f$precision = \frac{TN}{TN + FP}@f$
**/
float specificity_score_conf (float** confusion_matrix, int classes)
{
    if(classes == 2)
        return safe_division(confusion_matrix[0][0], confusion_matrix[0][0] + confusion_matrix[0][1]);

    int i, j, k;
    float mean = 0.0;

    for(i = 0; i < classes; ++i)
    {
        float num = 0.0, den = 0.0;

        for(j = 0; j < classes; ++j)
            for(k = 0; k < classes; ++k)
                num += confusion_matrix[j][k] * (i != j), den += confusion_matrix[j][k];

        mean += (num / den);
    }

    return mean / classes;
}


/** @brief Calculate the F score given a confusion matrix
 *  @details The F score is a way to weight both precision and recall for classification. It is a harmonic 
 *           mean,meaning that small values in either precision or recall will bring the F score down more 
 *           than average values in both.
 *              
 *           The general formula is:
 *          
 *           @f$$ (1 + \beta^2) * \frac{precision * recall}{\beta^2 * precision + recall} @f$$
 * 
 *           Where @f$\beta@f$ is a weighting constant between precision and recall, usually set to 1 (F1 score).
 * 
 *           Some more info can be found here https://en.wikipedia.org/wiki/F1_score  
 *  
 *  @param beta A weighting constant between precision and recall
**/
float f_score_conf (float** confusion_matrix, int classes, float beta)
{
    float precision = precision_score_conf(confusion_matrix, classes);
    float recall = recall_score_conf(confusion_matrix, classes);
    beta *= beta;

    return ((1 + beta) * precision * recall) / (beta * precision + recall);
}

/** @brief Calculate the F1 score given a confusion matrix
 *  @details Simply call f_score with @c beta set to 1
**/
float f1_score_conf (float** confusion_matrix, int classes)
{
    return f_score_conf(confusion_matrix, classes, 1.0);
}

//@}



float accuracy_score_pred (int* y_true, int* y_pred, int n, int classes)
{
    return prediction_metric(y_true, y_pred, n, classes, accuracy_score_conf);
}

float precision_score_pred (int* y_true, int* y_pred, int n, int classes)
{
    return prediction_metric(y_true, y_pred, n, classes, precision_score_conf);
}

float recall_score_pred (int* y_true, int* y_pred, int n, int classes)
{
    return prediction_metric(y_true, y_pred, n, classes, recall_score_conf);
}

float npv_score_pred (int* y_true, int* y_pred, int n, int classes)
{
    return prediction_metric(y_true, y_pred, n, classes, npv_score_conf);
}

float specificity_score_pred (int* y_true, int* y_pred, int n, int classes)
{
    return prediction_metric(y_true, y_pred, n, classes, specificity_score_conf);
}

float f1_score_pred (int* y_true, int* y_pred, int n, int classes)
{
    return prediction_metric(y_true, y_pred, n, classes, f1_score_conf);
}



float accuracy_score (int* y_true, float** y_score, int n, int classes)
{
    return confusion_matrix_metric(y_true, y_score, n, classes, accuracy_score_conf);
}

float precision_score (int* y_true, float** y_score, int n, int classes)
{
    return confusion_matrix_metric(y_true, y_score, n, classes, precision_score_conf);
}

float recall_score (int* y_true, float** y_score, int n, int classes)
{
    return confusion_matrix_metric(y_true, y_score, n, classes, recall_score_conf);
}

float npv_score (int* y_true, float** y_score, int n, int classes)
{
    return confusion_matrix_metric(y_true, y_score, n, classes, npv_score_conf);
}

float specificity_score (int* y_true, float** y_score, int n, int classes)
{
    return confusion_matrix_metric(y_true, y_score, n, classes, specificity_score_conf);
}

float f1_score (int* y_true, float** y_score, int n, int classes)
{
    return confusion_matrix_metric(y_true, y_score, n, classes, f1_score_conf);
}



/** @brief Calculate curves of different metrics given a vector of true classes and a vector of scores for binary classification
 *  @details Calculate the given metrics for each cutting point of the scores, that is, at every point where one observation changes its class.
 *           The returned vectors have actually <tt>n-1</tt> elements, considering the middle position between each consecutive element.
 *  @param y_true A int vector containing the value of the actual classes for each observation
 *  @param y_score A float vector containing the scores for the positive class
 *  @param n Size of the vectors
 *  @return A float matrix containing three vectors in its rows: the cutting point of the scores, the first and second metrics respectivelly
**/
//@{

/// precision-recall curve
float** precision_recall_curve (int* y_true, float* y_score, int n)
{
    int i;
    float true_pos = 0.0, true_neg = 0.0, false_pos = 0.0, false_neg = 0.0;

    float** curve = new_mat(3, n-1, sizeof(float));

    paired_sort(y_score, (float*)y_true, n);

    for(i = 0; i < n; ++i)
    {
        true_pos += y_true[i];
        false_pos += !y_true[i];
    }

    for(i = 0; i < n-1; ++i)
    {
        true_pos -= y_true[i];
        false_pos -= !y_true[i];
        false_neg += y_true[i];

        curve[0][i] = (y_score[i] + y_score[i+1]) / 2;
        curve[1][i] = true_pos / (true_pos + false_neg);
        curve[2][i] = true_pos / (true_pos + false_pos);
    }

    return curve;
}

/// roc curve
float** roc_curve (int* y_true, float* y_score, int n)
{
    int i;
    float true_pos = 0.0, true_neg = 0.0, false_pos = 0.0, false_neg = 0.0;

    float** curve = new_mat(3, n-1, sizeof(float));

    paired_sort(y_score, (float*)y_true, n);

    for(i = 0; i < n; ++i)
    {
        true_pos += y_true[i];
        false_pos += !y_true[i];
    }

    for(i = 0; i < n-1; ++i)
    {
        true_pos -= y_true[i];
        true_neg += !y_true[i];
        false_pos -= !y_true[i];
        false_neg += y_true[i];

        curve[0][i] = (y_score[i] + y_score[i+1]) / 2;
        curve[1][i] = true_pos / (true_pos + false_neg);
        curve[2][i] = 1.0 - true_neg / (true_neg + false_pos);
    }

    return curve;
}

/// npv_tnr curve (negative predictive value and true negative value)
float** npv_tnr_curve (int* y_true, float* y_score, int n)
{
    int i;
    float true_neg = 0.0, false_neg = 0.0, false_pos = 0.0;

    float** curve = new_mat(3, n-1, sizeof(float));

    paired_sort(y_score, (float*)y_true, n);

    for(i = 0; i < n; ++i)
        false_pos += !y_true[i];

    for(i = 0; i < n-1; ++i)
    {
        true_neg += !y_true[i];
        false_neg += y_true[i];
        false_pos -= !y_true[i];

        curve[0][i] = (y_score[i] + y_score[i+1]) / 2;
        curve[1][i] = true_neg / (true_neg + false_pos);
        curve[2][i] = true_neg / (true_neg + false_neg);
    }

    return curve;
}

//@}



float* fixed_metric_score (int* y_true, float* y_score, int n, float minimum_value, float** (*curve_function)(int*, float*, int))
{
    int i, pos = 0;
    float best_value = 0.0;
    float** curve = curve_function(y_true, y_score, n);

    for(i = 0; i < n; ++i)
    {
        if(curve[2][i] > best_value)
            best_value = curve[2][i], pos = i;

        if(curve[1][i] < minimum_value)
            break;
    }

    float* result = malloc(3 * sizeof(float));

    for(i = 0; i < 3; ++i)
        result[i] = curve[i][pos];

    del_mat(curve, 3);

    return result;
}

float* fixed_recall_score (int* y_true, float* y_score, int n, float minimum_value)
{
    return fixed_metric_score(y_true, y_score, n, minimum_value, precision_recall_curve);
}

float* fixed_specificity_score (int* y_true, float* y_score, int n, float minimum_value)
{
    return fixed_metric_score(y_true, y_score, n, minimum_value, npv_tnr_curve);
}
