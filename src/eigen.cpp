#include <algorithm>
#include <chrono>
#include <iostream>
#include "eigen.h"

#include <pybind11/pybind11.h>

using namespace std;

namespace py=pybind11;

bool esAutovector(Vector v, const Matrix& X, double lambda, double eps) {
    double norma = (X*v - lambda*v).norm();
    return norma < eps;
}

pair<double, Vector> power_iteration(const Matrix& X, unsigned num_iter, double eps)
{
    Vector b;
    double eigenvalue;
    double eps3 = eps*eps*eps;
    b = Vector::Random(X.cols());
    //No pusimos condicion de corte, no sabemos si recalcular el eigenvalue dentro del for cuesta mas
    //que hacer mas iteracion solo de lo que esta dentro del for
    for (unsigned int i = 0; i < num_iter; i++){
        Vector producto = X*b;
        Vector nuevoB = (producto / producto.norm());
        if((nuevoB - b).norm() < eps3) {
            b = nuevoB;
            break;
        }
        b = nuevoB;
    }
    eigenvalue = (b.transpose() * X).dot(b) / b.transpose().dot(b);

    return make_pair(eigenvalue, b / b.norm());
}

pair<Vector, Matrix> get_first_eigenvalues(const Matrix& X, unsigned num, unsigned num_iter, double epsilon)
{
    Matrix A(X);
    Vector eigvalues(num);
    Matrix eigvectors(A.rows(), num);

    for (unsigned int i = 0; i < num; i++){
        /* py::print("Obtengo el autovector-------------------------------------------------------------------------------------"); */
        pair<double, Vector> res = power_iteration(A, num_iter, epsilon);
        double lamda_i = res.first;
        Vector v_i = res.second;
        eigvalues(i) = lamda_i;
        eigvectors.col(i) = v_i;
        A = A - (lamda_i * v_i * v_i.transpose());
    }

    return make_pair(eigvalues, eigvectors);
}
