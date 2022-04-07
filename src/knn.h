#pragma once

#include "types.h"


class KNNClassifier {
public:
    KNNClassifier(unsigned int n_neighbors);

    void fit(Matrix X, Matrix y);

    Vector predict(Matrix X);
private:
    unsigned int k_neighbors;
    Matrix matrix_train;
    Matrix matrix_train_labels;
};
