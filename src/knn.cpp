#include <algorithm>
//#include <chrono>
#include <iostream>
#include "knn.h"
#include <pybind11/pybind11.h>

using namespace std;
namespace py=pybind11;

bool sortbysecasc(const pair<int,double> &a, const pair<int,double> &b) {
    return a.second < b.second;
}

KNNClassifier::KNNClassifier(unsigned int n_neighbors) {
    k_neighbors = n_neighbors;
}

void KNNClassifier::fit(Matrix X, Matrix y) {
    matrix_train = X;
    matrix_train_labels = y;
}


Vector KNNClassifier::predict(Matrix X)
{
    // Creamos vector columna a devolver
    auto ret = Vector(X.rows());

    for (unsigned k = 0; k < X.rows(); ++k) {
        Matrix train(matrix_train);

        vector<pair<int, double>> distances;

        train.rowwise() -= X.row(k);
        for (unsigned int i = 0; i < train.rows(); i++) {
            distances.push_back(make_pair(i, train.row(i).norm()));
        }

        sort(distances.begin(), distances.end(), sortbysecasc);
        
        map<int, int> votos;
        for (unsigned int i = 0; i < k_neighbors; i++) {
            int label = matrix_train_labels.row(distances[i].first)(0);
            if (votos.find(label) != votos.end()) {
                votos[label]++;
            } else {
                votos[label] = 1;
            }
        }

        pair<int, int> maxVoto = make_pair(0, 0);
        for (auto const& voto : votos) {
            if(voto.second > maxVoto.second) {
                maxVoto = voto;
            }
        }
        
        ret(k) = maxVoto.first;
    }

    return ret;
}
