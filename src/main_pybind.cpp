//
// Created by pachi on 5/6/19.
// Ejemplo basado en la pregunta de stackoverflow:
// https://stackoverflow.com/questions/47762543/segfault-when-using-pybind11-wrappers-in-c

#include <iostream>
#include "pca.h"
#include "eigen.h"

#include <pybind11/embed.h>

#include "types.h"

using namespace std;

namespace py = pybind11;

void eigenTester() {
  Matrix mat(6,6);
  mat <<  1, 2, 3, 4, 5, 6,
          2, 2, 3, 4, 5, 6,
          3, 3, 3, 4, 5, 6,
          4, 4, 4, 4, 4, 6,
          5, 5, 5, 4, 5, 6,
          6, 6, 6, 6, 6, 6;

  pair<Vector, Matrix> res = get_first_eigenvalues(mat, 3, 100, 10e-6);

  cout << "Autovalores" << endl;
  cout << res.first << endl;

  cout << "Autovectores" << endl;
  cout << res.second << endl;

  cout << "------------------------------------------" << endl;

  auto eigenvalues = mat.eigenvalues();
  cout << "Autovalores" << endl;
  cout << eigenvalues << endl;

  cout << "------------------------------------------" << endl;

  Matrix algo = mat * res.second;
  algo.col(0) = algo.col(0) / res.first(0);
  algo.col(1) = algo.col(1) / res.first(1);
  algo.col(2) = algo.col(2) / res.first(2);

  cout << algo << endl;

  /*for (int i = 0; i < res.first.size(); i++) {
    cout << "Autovalor: " << res.first(i) << endl;
  }

  for (int i = 0; i < res.second.cols(); i++) {
    cout << "Autovector: " << endl;
    for (int j = 0; j < res.second.col(i).rows(); j++) {
      cout << res.second.col(i).row(j) << " ";
    }
    cout << endl;
  }*/
  
}


int main(int argc, char** argv){

  py::scoped_interpreter guard{};

  py::print("Hola pybind!");

  std::cout << "Hola mundo!" << std::endl;
  eigenTester();
  std::cout << "Chau mundo!" << std::endl;

  return 0;
}