//
// Created by andre on 4/4/22.
//

#ifndef STRASSENCUDA__MATRIX_H_
#define STRASSENCUDA__MATRIX_H_

#include <vector>

template<typename T>
struct Matrix{
  unsigned long width;
  unsigned long height;
  std::vector<T> elements;
};

template<typename T>
struct CUDAMatrix{
  unsigned long width;
  unsigned long height;
  T *elements;
};

#endif //STRASSENCUDA__MATRIX_H_
