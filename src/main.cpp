//
// Created by pachi on 5/6/19.
//

#include <iostream>
#include "pca.h"
#include "eigen.h"

#include "types.h"

using namespace std;

int main(int argc, char** argv){

  std::cout << "Hola mundo!" << std::endl;
  eigenTester();
  std::cout << "Chau mundo!" << std::endl;
  return 0;
}

void eigenTester() {
  Matrix mat(6,6);
  mat <<  1, 2, 3, 4, 5, 6,
          2, 2, 3, 4, 5, 6,
          3, 3, 3, 4, 5, 6,
          4, 4, 4, 4, 4, 6,
          5, 5, 5, 4, 5, 6,
          6, 6, 6, 6, 6, 6;

  pair<Vector, Matrix> res = get_first_eigenvalues(mat, 10, 100, 10e-6);

  for (int i = 0; i < res.first.size(); i++) {
    cout << "Autovalor: " << res.first(i);
  }

  for (int i = 0; i < res.second.size(); i++) {
    cout << "Autovector: " << endl;
    for (int j = 0; j < res.second.row(i).size(); j++) {
      cout << res.second.row(i).col(j) << " ";
    }
    cout << endl;
  }
  
}
