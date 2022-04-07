#include <iostream>
#include "pca.h"
#include "eigen.h"

using namespace std;


PCA::PCA(unsigned int n_components) {
  components_a_reducir = n_components;
}

void PCA::fit(Matrix X) {
  Matrix x_train = X;
  double divisor = sqrt(x_train.rows() - 1); 
  //Calculando la media de cada columna y restandosela a la matriz X
  auto v = x_train.colwise().mean();
  x_train = x_train.rowwise() - v;
  x_train = x_train / divisor;
 
  Matrix M_x = x_train.transpose() * x_train;

  //TODO: Analizar si este 10000 y 10e-6 deberian ser parametros del metodo.
  pair<Vector, Matrix> res = get_first_eigenvalues(M_x, components_a_reducir, 10000, 10e-6);

  transformation = res.second;
}


MatrixXd PCA::transform(Matrix X) {
  return X * transformation;
}
