#ifndef PARSER_H
#define PARSER_H
#include "darknet.h"
#include "network.h"

typedef struct{
    char *type;
    list *options;
}section;

typedef enum { ACCURACY, F1_SCORE, LOSS } TRAIN_METRIC;

typedef struct
{
    TRAIN_METRIC metric;
    int eval_epochs;
    int max_epochs;
    int patience;
    int seed;
    char* training_log;

} SSM_Params;

void save_network(network net, char *filename);
void save_weights_double(network net, char *filename);

#endif
