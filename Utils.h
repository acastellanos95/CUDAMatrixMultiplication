//
// Created by andre on 4/4/22.
//

#ifndef STRASSENCUDA__UTILS_H_
#define STRASSENCUDA__UTILS_H_

#include <vector>
#include <random>
#include <type_traits>

template<typename T>
std::vector<T> flatMatrix(std::vector<std::vector<T>> &matrix) {
  auto result = std::vector<T>();
  for (auto &row: matrix) {
    for (auto &element: row) {
      result.push_back(element);
    }
  }
  return result;
}

template<typename T>
void initMatrix(std::vector<std::vector<T>> &A){
  if(std::is_integral<T>::value){
    std::random_device rd;
    std::default_random_engine e( rd() );
    std::uniform_int_distribution<long> uniform_dist(0, 5);
    for(size_t indexRow = 0; indexRow < A.size(); ++indexRow){
      for (size_t indexCol = 0; indexCol < A[indexRow].size(); ++indexCol) {
        A[indexRow][indexCol] = uniform_dist(e);
      }
    }
  } else if(std::is_floating_point<T>::value){
    std::random_device rd;
    std::default_random_engine e( rd() );
    std::uniform_real_distribution<double> uniform_dist(0, 5);
    for(size_t indexRow = 0; indexRow < A.size(); ++indexRow){
      for (size_t indexCol = 0; indexCol < A[indexRow].size(); ++indexCol) {
        A[indexRow][indexCol] = uniform_dist(e);
      }
    }
  }
}

template<typename T>
std::vector<std::vector<T>> add(const std::vector<std::vector<T>> &A, const std::vector<std::vector<T>> &B) {
  std::vector<std::vector<T>> result(A.size(), std::vector<T>(A[0].size()));
  for (size_t row = 0; row < A.size(); ++row) {
    for (size_t col = 0; col < A[0].size(); ++col) {
      result[row][col] = A[row][col] + B[row][col];
    }
  }
  return result;
}

template<typename T>
std::vector<std::vector<T>> subtract(const std::vector<std::vector<T>> &A, const std::vector<std::vector<T>> &B) {
  std::vector<std::vector<T>> result(A.size(), std::vector<T>(A[0].size()));
  for (size_t row = 0; row < A.size(); ++row) {
    for (size_t col = 0; col < A[0].size(); ++col) {
      result[row][col] = A[row][col] - B[row][col];
    }
  }
  return result;
}

template<typename T>
std::vector<std::vector<T>> parallelAdd(const std::vector<std::vector<T>> &A, const std::vector<std::vector<T>> &B){
  std::vector<std::vector<T>> C = std::vector<std::vector<T>>(A.size(), std::vector<T>(B.size(), 0));
  if(A.size() == B.size() && A[0].size() == B[0].size()){
#pragma omp parallel for
    for (size_t i = 0; i < A.size(); i++) {
      for (size_t j = 0; j < B.size(); j++) {
        C[i][j] = A[i][j] + B[i][j];
      }
    }
  } else {
    throw std::length_error("No coinciden el número de columnas ni filas");
  }
  return C;
}

template<typename T>
std::vector<std::vector<T>> parallelSubstract(const std::vector<std::vector<T>> &A, const std::vector<std::vector<T>> &B){
  std::vector<std::vector<T>> C = std::vector<std::vector<T>>(A.size(), std::vector<T>(B.size(), 0));
  if(A.size() == B.size() && A[0].size() == B[0].size()){
#pragma omp parallel for
    for (size_t i = 0; i < A.size(); i++) {
      for (size_t j = 0; j < B.size(); j++) {
        C[i][j] = A[i][j] - B[i][j];
      }
    }
  } else {
    throw std::length_error("No coinciden el número de columnas ni filas");
  }
  return C;
}

template<typename T>
std::vector<std::vector<T>> multiplicationNormal(const std::vector<std::vector<T>> &A, const std::vector<std::vector<T>> &B) {
  std::vector<std::vector<T>> result(A.size(), std::vector<T>(B[0].size(), 0));
  if (A[0].size() != B.size())
    throw std::length_error("number of columns of A don't match number of rows in B");
  for (size_t row = 0; row < A.size(); ++row) {
    for (size_t col = 0; col < B[0].size(); ++col) {
      for (size_t multIndex = 0; multIndex < B.size(); ++multIndex) {
        result[row][col] += A[row][multIndex] * B[multIndex][col];
      }
    }
  }
  return result;
}

template<typename T>
std::vector<std::vector<T>> parallelMultiplication(const std::vector<std::vector<T>> &A, const std::vector<std::vector<T>> &B){
  auto C = std::vector<std::vector<T>>(A.size(), std::vector<T>(B[0].size(), 0));
  if(A[0].size() == B.size()){
#pragma omp parallel for
    for (size_t i = 0; i < A.size(); i++) {
      for (size_t j = 0; j < B[0].size(); j++) {
        for (size_t k = 0; k < B.size(); k++) {
          C[i][j] += A[i][k]*B[k][j];
        }
      }
    }
  } else {
    throw std::length_error("El número de columnas de A no es igual al número de filas de B");
  }
  return C;
}

template<typename T>
void normalStrassenMultiplication(const std::vector<std::vector<T>> &A, const std::vector<std::vector<T>> &B, std::vector<std::vector<T>> &C, const int &threshold){
  if(A.size() <= threshold){
    C = multiplicationNormal(A,B);
  } else{
    /*Nuevo tamaño*/
    int newSize = A.size()/2;

    /*Particiones*/
    std::vector<std::vector<T>> A11(newSize, std::vector<T>(newSize)), A12(newSize, std::vector<T>(newSize)), A21(newSize, std::vector<T>(newSize)), A22(newSize, std::vector<T>(newSize)),
        B11(newSize, std::vector<T>(newSize)), B12(newSize, std::vector<T>(newSize)), B21(newSize, std::vector<T>(newSize)), B22(newSize, std::vector<T>(newSize)),
        C11(newSize, std::vector<T>(newSize)), C12(newSize, std::vector<T>(newSize)), C21(newSize, std::vector<T>(newSize)), C22(newSize, std::vector<T>(newSize)),
    /*resultados intermedios*/
    A_intermedio(newSize, std::vector<T>(newSize)), B_intermedio(newSize, std::vector<T>(newSize)),
    /*Matrices I,II,...,VII*/
    M1(newSize, std::vector<T>(newSize)), M2(newSize, std::vector<T>(newSize)), M3(newSize, std::vector<T>(newSize)),
        M4(newSize, std::vector<T>(newSize)), M5(newSize, std::vector<T>(newSize)), M6(newSize, std::vector<T>(newSize)),
        M7(newSize, std::vector<T>(newSize));

    /*asignar A y B a sus respectivos cuadrantes*/
    for (int i=0; i<newSize; i++){
      for (int j=0; j<newSize; j++){
        A11[i][j] = A[i][j];
        A12[i][j] = A[i][j+newSize];
        A21[i][j] = A[i+newSize][j];
        A22[i][j] = A[i+newSize][j+newSize];
        B11[i][j] = B[i][j];
        B12[i][j] = B[i][j+newSize];
        B21[i][j] = B[i+newSize][j];
        B22[i][j] = B[i+newSize][j+newSize];
      }
    }

    //computar submatrices
    //M1 = (A11+A22)*(B11+B22)
    A_intermedio = add(A11, A22);
    B_intermedio = add(B11, B22);
    normalStrassenMultiplication(A_intermedio, B_intermedio, M1, threshold);

    //M2 = (A21+A22)*B11
    A_intermedio = add(A21, A22);
    normalStrassenMultiplication(A_intermedio, B11, M2, threshold);

    //M3 = A11(B12-B22)
    B_intermedio = subtract(B12, B22);
    normalStrassenMultiplication(A11, B_intermedio, M3, threshold);

    //M4 = A22(B21-B11)
    B_intermedio = subtract(B21, B11);
    normalStrassenMultiplication(A22, B_intermedio, M4, threshold);

    //M5 = (A11+A12)B22
    A_intermedio = add(A11, A12);
    normalStrassenMultiplication(A_intermedio, B22, M5, threshold);

    //M6 = (A21-A11)(B11+B12)
    A_intermedio = subtract(A21, A11);
    B_intermedio = add(B11, B12);
    normalStrassenMultiplication(A_intermedio, B_intermedio, M6, threshold);

    //M7 = (A12-A22)(B21+B22)
    A_intermedio = subtract(A12, A22);
    B_intermedio = add(B21, B22);
    normalStrassenMultiplication(A_intermedio, B_intermedio, M7, threshold);

    //Calcular resultados de C
    //C11 = M1+M4-M5+M7
    A_intermedio = add(M1, M4);
    B_intermedio = subtract(A_intermedio, M5);
    C11 = add(B_intermedio, M7);

    //C12 = M3+M5
    C12 = add(M3, M5);

    //C21 = M2+M4
    C21 = add(M2, M4);

    //C22 = M1-M2+M3+M6
    A_intermedio = subtract(M1, M2);
    B_intermedio = add(A_intermedio, M3);
    C22 = add(B_intermedio, M6);

    //calcular resultado final
    for(int i=0; i<newSize; i++){
      for(int j=0; j<newSize; j++){
        C[i][j] = C11[i][j];
        C[i][newSize+j] = C12[i][j];
        C[newSize+i][j] = C21[i][j];
        C[newSize+i][newSize+j] = C22[i][j];
      }
    }
  }
}

template<typename T>
void parallelStrassenMultiplication(const std::vector<std::vector<T>> &A, const std::vector<std::vector<T>> &B, std::vector<std::vector<T>> &C, const int &threshold){
  if(A.size() <= threshold){
    C = parallelMultiplication(A,B);
  } else{
    /*Nuevo tamaño*/
    int newSize = A.size()/2;

    /*Particiones*/
    std::vector<std::vector<T>> A11(newSize, std::vector<T>(newSize)), A12(newSize, std::vector<T>(newSize)), A21(newSize, std::vector<T>(newSize)), A22(newSize, std::vector<T>(newSize)),
        B11(newSize, std::vector<T>(newSize)), B12(newSize, std::vector<T>(newSize)), B21(newSize, std::vector<T>(newSize)), B22(newSize, std::vector<T>(newSize)),
        C11(newSize, std::vector<T>(newSize)), C12(newSize, std::vector<T>(newSize)), C21(newSize, std::vector<T>(newSize)), C22(newSize, std::vector<T>(newSize)),
    /*resultados intermedios*/
    A_intermedio(newSize, std::vector<T>(newSize)), B_intermedio(newSize, std::vector<T>(newSize)),
    /*Matrices I,II,...,VII*/
    M1(newSize, std::vector<T>(newSize)), M2(newSize, std::vector<T>(newSize)), M3(newSize, std::vector<T>(newSize)),
        M4(newSize, std::vector<T>(newSize)), M5(newSize, std::vector<T>(newSize)), M6(newSize, std::vector<T>(newSize)),
        M7(newSize, std::vector<T>(newSize));

    /*asignar A y B a sus respectivos cuadrantes*/
    for (int i=0; i<newSize; i++){
      for (int j=0; j<newSize; j++){
        A11[i][j] = A[i][j];
        A12[i][j] = A[i][j+newSize];
        A21[i][j] = A[i+newSize][j];
        A22[i][j] = A[i+newSize][j+newSize];
        B11[i][j] = B[i][j];
        B12[i][j] = B[i][j+newSize];
        B21[i][j] = B[i+newSize][j];
        B22[i][j] = B[i+newSize][j+newSize];
      }
    }

    //computar submatrices
    //M1 = (A11+A22)*(B11+B22)
    A_intermedio = parallelAdd(A11, A22);
    B_intermedio = parallelAdd(B11, B22);
    parallelStrassenMultiplication(A_intermedio, B_intermedio, M1, threshold);

    //M2 = (A21+A22)*B11
    A_intermedio = parallelAdd(A21, A22);
    parallelStrassenMultiplication(A_intermedio, B11, M2, threshold);

    //M3 = A11(B12-B22)
    B_intermedio = parallelSubstract(B12, B22);
    parallelStrassenMultiplication(A11, B_intermedio, M3, threshold);

    //M4 = A22(B21-B11)
    B_intermedio = parallelSubstract(B21, B11);
    parallelStrassenMultiplication(A22, B_intermedio, M4, threshold);

    //M5 = (A11+A12)B22
    A_intermedio = parallelAdd(A11, A12);
    parallelStrassenMultiplication(A_intermedio, B22, M5, threshold);

    //M6 = (A21-A11)(B11+B12)
    A_intermedio = parallelSubstract(A21, A11);
    B_intermedio = parallelAdd(B11, B12);
    parallelStrassenMultiplication(A_intermedio, B_intermedio, M6, threshold);

    //M7 = (A12-A22)(B21+B22)
    A_intermedio = parallelSubstract(A12, A22);
    B_intermedio = parallelAdd(B21, B22);
    parallelStrassenMultiplication(A_intermedio, B_intermedio, M7, threshold);

    //Calcular resultados de C
    //C11 = M1+M4-M5+M7
    A_intermedio = parallelAdd(M1, M4);
    B_intermedio = parallelSubstract(A_intermedio, M5);
    C11 = parallelAdd(B_intermedio, M7);

    //C12 = M3+M5
    C12 = parallelAdd(M3, M5);

    //C21 = M2+M4
    C21 = parallelAdd(M2, M4);

    //C22 = M1-M2+M3+M6
    A_intermedio = parallelSubstract(M1, M2);
    B_intermedio = parallelAdd(A_intermedio, M3);
    C22 = parallelAdd(B_intermedio, M6);

    //calcular resultado final
    for(int i=0; i<newSize; i++){
      for(int j=0; j<newSize; j++){
        C[i][j] = C11[i][j];
        C[i][newSize+j] = C12[i][j];
        C[newSize+i][j] = C21[i][j];
        C[newSize+i][newSize+j] = C22[i][j];
      }
    }
  }
}

template<typename T>
__global__ void multiplication(CUDAMatrix<T> matrix1, CUDAMatrix<T> matrix2, CUDAMatrix<T> result) {
  unsigned int row = threadIdx.y + (blockDim.y * blockIdx.y);
  unsigned int col = threadIdx.x + (blockDim.x * blockIdx.x);
  T tmp, rb, rc;
  if (row < matrix1.height && col < matrix2.width) {
    tmp = 0;
    for (size_t multIndex = 0; multIndex < matrix1.width; ++multIndex) {
      rb = matrix1.elements[row * matrix1.width + multIndex];
      rc = matrix2.elements[multIndex * matrix2.width + col];
      tmp += rb * rc;
    }
    result.elements[row * result.width + col] = tmp;
  }
}

template<typename T>
__global__ void multiplicationShrd(CUDAMatrix<T> matrix1, CUDAMatrix<T> matrix2, CUDAMatrix<T> result) {
  // Global row and column of result matrix
  int rowMatrix = threadIdx.y + blockIdx.y * blockDim.y;
  int colMatrix = threadIdx.x + blockIdx.x * blockDim.x;
  // Block indexing of rows and columns
  int subRow = threadIdx.y;
  int subCol = threadIdx.x;

  unsigned int k, K;
  // Number of NT rows or columns to advance to get to the matrix1.width or matrix2.height
  unsigned int L = (matrix2.height % NT == 0) ? (matrix2.height / NT) : (matrix2.height / NT + 1);
  T tmp = 0 ;

  // Shared tiled square
  __shared__ T subMatrix1[NT][NT], subMatrix2[NT][NT];

  for (K = 0; K < L; K++) {
    // select matrix1 and matrix2 squared tile from global col and row for K square
    subMatrix1[subRow][subCol] = rowMatrix < result.height && K * NT + subCol < matrix1.width ? matrix1.elements[rowMatrix * matrix1.width + (K * NT + subCol)] : 0;
    subMatrix2[subRow][subCol] = colMatrix < result.width && K * NT + subRow < matrix2.height ? matrix2.elements[(K * NT + subRow) * matrix2.width + colMatrix] : 0;

    // sync to have all scratchpad full
    __syncthreads();
    for (k = 0; k < NT; k++) {
      tmp += subMatrix1[subRow][k] * subMatrix2[k][subCol];
    }
    // sync to don't get wrong submatrices in next iteration
    __syncthreads();
  }

  // write to result
  if (colMatrix < result.width && rowMatrix < result.height)
    result.elements[rowMatrix * result.width + colMatrix] = tmp;
}

template<typename T>
__global__ void addCuda(CUDAMatrix<T> matrix1, CUDAMatrix<T> matrix2, CUDAMatrix<T> result) {
  unsigned int row = threadIdx.y + (blockDim.y * blockIdx.y);
  unsigned int col = threadIdx.x + (blockDim.x * blockIdx.x);

  if (matrix1.height != matrix2.height && matrix2.width != matrix1.width) {
    asm("trap;");
  } else {
    if (row < matrix1.height && col < matrix1.width) {
      result.elements[row * matrix1.width + col] =
          matrix1.elements[row * matrix1.width + col] + matrix2.elements[row * matrix1.width + col];
    }
  }
}

template<typename T>
__global__ void subtractCuda(CUDAMatrix<T> matrix1, CUDAMatrix<T> matrix2, CUDAMatrix<T> result) {
  unsigned int row = threadIdx.y + (blockDim.y * blockIdx.y);
  unsigned int col = threadIdx.x + (blockDim.x * blockIdx.x);

  if (matrix1.height != matrix2.height && matrix2.width != matrix1.width) {
    asm("trap;");
  } else {
    if (row < matrix1.height && col < matrix1.width) {
      result.elements[row * matrix1.width + col] =
          matrix1.elements[row * matrix1.width + col] - matrix2.elements[row * matrix1.width + col];
    }
  }
}

template<typename T>
void addCudaWrapper(Matrix<T> &matrix1, Matrix<T> &matrix2, Matrix<T> &result){
  CUDAMatrix<T> d_matrix1, d_matrix2, d_matrixRes;
  d_matrix1.height = matrix1.height;
  d_matrix2.height = matrix2.height;
  d_matrixRes.height = result.height;
  d_matrix1.width = matrix1.width;
  d_matrix2.width = matrix2.width;
  d_matrixRes.width = result.width;
  cudaMalloc(&d_matrix1.elements, d_matrix1.height * d_matrix1.width * sizeof(T));
  cudaMalloc(&d_matrix2.elements, d_matrix2.height * d_matrix2.width * sizeof(T));
  cudaMalloc(&d_matrixRes.elements, d_matrixRes.height * d_matrixRes.width * sizeof(T));
  cudaMemcpy(d_matrix1.elements, matrix1.elements.data(), d_matrix1.height * d_matrix1.width * sizeof(T),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_matrix2.elements, matrix2.elements.data(), d_matrix2.height * d_matrix2.width * sizeof(T),
             cudaMemcpyHostToDevice);
  dim3 dimBlock(NT, NT);
  dim3 dimGrid((d_matrixRes.width + dimBlock.x - 1) / dimBlock.x, (d_matrixRes.height + dimBlock.y - 1) / dimBlock.y);
  addCuda<<<dimGrid, dimBlock>>>(d_matrix1, d_matrix2, d_matrixRes);
  cudaDeviceSynchronize();
  T *elements = new T[result.width*result.height];
  cudaMemcpy(elements, d_matrixRes.elements, result.width*result.height*sizeof(T), cudaMemcpyDeviceToHost);
  result.elements = std::vector<T>();
  result.elements.insert(result.elements.begin(), elements, elements + (result.height * result.width));
  cudaFree(d_matrix1.elements);
  cudaFree(d_matrix2.elements);
  cudaFree(d_matrixRes.elements);
  free(elements);
}

template<typename T>
void substractCudaWrapper(Matrix<T> &matrix1, Matrix<T> &matrix2, Matrix<T> &result){
  CUDAMatrix<T> d_matrix1, d_matrix2, d_matrixRes;
  d_matrix1.height = matrix1.height;
  d_matrix2.height = matrix2.height;
  d_matrixRes.height = result.height;
  d_matrix1.width = matrix1.width;
  d_matrix2.width = matrix2.width;
  d_matrixRes.width = result.width;
  cudaMalloc(&d_matrix1.elements, d_matrix1.height * d_matrix1.width * sizeof(T));
  cudaMalloc(&d_matrix2.elements, d_matrix2.height * d_matrix2.width * sizeof(T));
  cudaMalloc(&d_matrixRes.elements, d_matrixRes.height * d_matrixRes.width * sizeof(T));
  cudaMemcpy(d_matrix1.elements, matrix1.elements.data(), d_matrix1.height * d_matrix1.width * sizeof(T),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_matrix2.elements, matrix2.elements.data(), d_matrix2.height * d_matrix2.width * sizeof(T),
             cudaMemcpyHostToDevice);
  dim3 dimBlock(NT, NT);
  dim3 dimGrid((d_matrixRes.width + dimBlock.x - 1) / dimBlock.x, (d_matrixRes.height + dimBlock.y - 1) / dimBlock.y);
  subtractCuda<<<dimGrid, dimBlock>>>(d_matrix1, d_matrix2, d_matrixRes);
  cudaDeviceSynchronize();
  T *elements = new T[result.width*result.height];
  cudaMemcpy(elements, d_matrixRes.elements, result.width*result.height*sizeof(T), cudaMemcpyDeviceToHost);
  result.elements = std::vector<T>();
  result.elements.insert(result.elements.begin(), elements, elements + (result.height * result.width));
  cudaFree(d_matrix1.elements);
  cudaFree(d_matrix2.elements);
  cudaFree(d_matrixRes.elements);
  free(elements);
}

template<typename T>
void multiplicationCudaWrapper(Matrix<T> &matrix1, Matrix<T> &matrix2, Matrix<T> &result){
  CUDAMatrix<T> d_matrix1, d_matrix2, d_matrixRes;
  d_matrix1.height = matrix1.height;
  d_matrix2.height = matrix2.height;
  d_matrixRes.height = matrix1.height;
  d_matrix1.width = matrix1.width;
  d_matrix2.width = matrix2.width;
  d_matrixRes.width = matrix2.width;
  cudaMalloc(&d_matrix1.elements, d_matrix1.height * d_matrix1.width * sizeof(T));
  cudaMalloc(&d_matrix2.elements, d_matrix2.height * d_matrix2.width * sizeof(T));
  cudaMalloc(&d_matrixRes.elements, d_matrixRes.height * d_matrixRes.width * sizeof(T));
  cudaMemcpy(d_matrix1.elements, matrix1.elements.data(), d_matrix1.height * d_matrix1.width * sizeof(T),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_matrix2.elements, matrix2.elements.data(), d_matrix2.height * d_matrix2.width * sizeof(T),
             cudaMemcpyHostToDevice);
  dim3 dimBlock(NT, NT);
  dim3 dimGrid((d_matrixRes.width + dimBlock.x - 1) / dimBlock.x, (d_matrixRes.height + dimBlock.y - 1) / dimBlock.y);
  multiplicationShrd<<<dimGrid, dimBlock>>>(d_matrix1, d_matrix2, d_matrixRes);
  cudaDeviceSynchronize();
  T *elements = new T[result.width*result.height];
  cudaMemcpy(elements, d_matrixRes.elements, result.width*result.height*sizeof(T), cudaMemcpyDeviceToHost);
  result.elements = std::vector<T>();
  result.elements.insert(result.elements.begin(), elements, elements + (result.height * result.width));
  cudaFree(d_matrix1.elements);
  cudaFree(d_matrix2.elements);
  cudaFree(d_matrixRes.elements);
  free(elements);
}

template<typename T>
void CUDAStrassenMultiplication(Matrix<T> &A, Matrix<T> &B, Matrix<T> &C, const int &threshold){
  if(A.height <= threshold){
    multiplicationCudaWrapper(A,B, C);
  } else{
    /*Nuevo tamaño*/
    unsigned long newSize = A.height/2;

    /*Particiones*/
    Matrix<T> A11, A12, A21, A22, B11, B12, B21, B22, C11, C12, C21, C22, M1, M2, M3, M4, M5, M6, M7, A_intermedio, B_intermedio;
    A11.width = A11.height = A12.width = A12.height = A21.width = A21.height = A22.width = A22.height = newSize;
    B11.width = B11.height = B12.width = B12.height = B21.width = B21.height = B22.width = B22.height = newSize;
    C11.width = C11.height = C12.width = C12.height = C21.width = C21.height = C22.width = C22.height = newSize;
    M1.width = M1.height = M2.width = M2.height = M3.width = M3.height = M4.width = M4.height = newSize;
    M5.width = M5.height = M6.width = M6.height = M7.width = M7.height = newSize;
    A_intermedio.width = A_intermedio.height = B_intermedio.width = B_intermedio.height = newSize;
    A11.elements = std::vector<T>(newSize*newSize);
    A12.elements = std::vector<T>(newSize*newSize);
    A21.elements = std::vector<T>(newSize*newSize);
    A22.elements = std::vector<T>(newSize*newSize);
    B11.elements = std::vector<T>(newSize*newSize);
    B12.elements = std::vector<T>(newSize*newSize);
    B21.elements = std::vector<T>(newSize*newSize);
    B22.elements = std::vector<T>(newSize*newSize);
    C11.elements = std::vector<T>(newSize*newSize);
    C12.elements = std::vector<T>(newSize*newSize);
    C21.elements = std::vector<T>(newSize*newSize);
    C22.elements = std::vector<T>(newSize*newSize);
    M1.elements = std::vector<T>(newSize*newSize);
    M2.elements = std::vector<T>(newSize*newSize);
    M3.elements = std::vector<T>(newSize*newSize);
    M4.elements = std::vector<T>(newSize*newSize);
    M5.elements = std::vector<T>(newSize*newSize);
    M6.elements = std::vector<T>(newSize*newSize);
    M7.elements = std::vector<T>(newSize*newSize);
    A_intermedio.elements = std::vector<T>(newSize*newSize);
    B_intermedio.elements = std::vector<T>(newSize*newSize);

    /*asignar A y B a sus respectivos cuadrantes*/
    for (int i=0; i<newSize; i++){
      for (int j=0; j<newSize; j++){
        A11.elements[i*A11.width + j] = A.elements[i*A.width + j];
        A12.elements[i*A12.width + j] = A.elements[i*A.width + j+newSize];
        A21.elements[i*A21.width + j] = A.elements[(i+newSize)*A.width + j];
        A22.elements[i*A22.width + j] = A.elements[(i+newSize)*A.width + j+newSize];
        B11.elements[i*B11.width + j] = B.elements[i*B.width + j];
        B12.elements[i*B12.width + j] = B.elements[i*B.width + j+newSize];
        B21.elements[i*B21.width + j] = B.elements[(i+newSize)*B.width + j];
        B22.elements[i*B22.width + j] = B.elements[(i+newSize)*B.width + j+newSize];
      }
    }

    //computar submatrices
    //M1 = (A11+A22)*(B11+B22)
    addCudaWrapper(A11, A22, A_intermedio);
    addCudaWrapper(B11, B22, B_intermedio);
    CUDAStrassenMultiplication(A_intermedio, B_intermedio, M1, threshold);

    //M2 = (A21+A22)*B11
    addCudaWrapper(A21, A22, A_intermedio);
    CUDAStrassenMultiplication(A_intermedio, B11, M2, threshold);

    //M3 = A11(B12-B22)
    substractCudaWrapper(B12, B22, B_intermedio);
    CUDAStrassenMultiplication(A11, B_intermedio, M3, threshold);

    //M4 = A22(B21-B11)
    substractCudaWrapper(B21, B11, B_intermedio);
    CUDAStrassenMultiplication(A22, B_intermedio, M4, threshold);

    //M5 = (A11+A12)B22
    addCudaWrapper(A11, A12, A_intermedio);
    CUDAStrassenMultiplication(A_intermedio, B22, M5, threshold);

    //M6 = (A21-A11)(B11+B12)
    substractCudaWrapper(A21, A11, A_intermedio);
    addCudaWrapper(B11, B12, B_intermedio);
    CUDAStrassenMultiplication(A_intermedio, B_intermedio, M6, threshold);

    //M7 = (A12-A22)(B21+B22)
    substractCudaWrapper(A12, A22, A_intermedio);
    addCudaWrapper(B21, B22, B_intermedio);
    CUDAStrassenMultiplication(A_intermedio, B_intermedio, M7, threshold);

    //Calcular resultados de C
    //C11 = M1+M4-M5+M7
    addCudaWrapper(M1, M4, A_intermedio);
    substractCudaWrapper(A_intermedio, M5, B_intermedio);
    addCudaWrapper(B_intermedio, M7, C11);

    //C12 = M3+M5
    addCudaWrapper(M3, M5, C12);

    //C21 = M2+M4
    addCudaWrapper(M2, M4, C21);

    //C22 = M1-M2+M3+M6
    substractCudaWrapper(M1, M2, A_intermedio);
    addCudaWrapper(A_intermedio, M3, B_intermedio);
    addCudaWrapper(B_intermedio, M6, C22);

    //calcular resultado final
    for(int i=0; i<newSize; i++){
      for(int j=0; j<newSize; j++){
        C.elements[i*C.width + j] = C11.elements[i*C11.width + j];
        C.elements[i*C.width + newSize+j] = C12.elements[i*C12.width + j];
        C.elements[(newSize+i)*C.width + j] = C21.elements[i*C21.width + j];
        C.elements[(newSize+i)*C.width + newSize+j] = C22.elements[i*C22.width + j];
      }
    }
  }
}

#endif //STRASSENCUDA__UTILS_H_
