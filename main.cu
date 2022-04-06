#include <iostream>
#include <fstream>
#include <omp.h>
#define NT 16
#include "lib/VariadicTable.h"
#include "Matrix.h"
#include "Utils.h"

int main() {
  unsigned long begin = 8;
  unsigned long end = 12;
  auto threshold = 128;
  std::ofstream file("reporte.txt", std::ofstream::out | std::ofstream::trunc);
  for(;begin <= end; ++begin){
    file << "dimension: " << (1<<begin) << ", threshold: " << threshold << '\n';
    VariadicTable<std::string, double, float> vt({"Algoritmo", "Tiempo (s)", "Error to normal"});
    auto matrix1 = std::vector<std::vector<float>>((1<<begin),std::vector<float>((1<<begin), 0.0));
    auto matrix2 = std::vector<std::vector<float>>((1<<begin),std::vector<float>((1<<begin), 0.0));
    initMatrix<float>(matrix1);
    initMatrix<float>(matrix2);

    // -------------------------------------- NORMAL & OMP --------------------------------------
    // Normal multiplication
    auto ti = omp_get_wtime();
    auto normalResult = multiplicationNormal(matrix1,matrix2);
    auto tf = omp_get_wtime();
    auto normalTimeSeconds = tf - ti;
    std::cout << "Tiempo normal: " << std::to_string(normalTimeSeconds) << '\n';
    vt.addRow("Normal", normalTimeSeconds, 0.0);

    // Normal Strassen multiplication
    std::vector<std::vector<float>> normalStrassenResult((1<<begin),std::vector<float>((1<<begin), 0.0));
    ti = omp_get_wtime();
    normalStrassenMultiplication(matrix1,matrix2, normalStrassenResult, threshold);
    tf = omp_get_wtime();
    auto normalStrassenTimeSeconds = tf - ti;
    std::cout << "Tiempo strassen normal: " << std::to_string(normalStrassenTimeSeconds) << '\n';
    float maxError = 0.0;
    for (size_t rowIndex = 0; rowIndex < normalStrassenResult.size(); ++rowIndex) {
      for (size_t colIndex = 0; colIndex < normalStrassenResult[0].size(); ++colIndex) {
        maxError = std::max(std::abs(normalResult[rowIndex][colIndex] - normalStrassenResult[rowIndex][colIndex]), maxError);
      }
    }
    std::cout << "Mayor error: " << std::to_string(maxError) << '\n';
    vt.addRow("Seq Strassen", normalStrassenTimeSeconds, maxError);

    // OMP multiplication
    ti = omp_get_wtime();
    auto OMPResult = parallelMultiplication(matrix1, matrix2);
    tf = omp_get_wtime();
    auto ompTimeSeconds = tf - ti;
    std::cout << "Tiempo multiplicación OMP: " << std::to_string(ompTimeSeconds) << '\n';
    maxError = 0.0;
    for (size_t rowIndex = 0; rowIndex < OMPResult.size(); ++rowIndex) {
      for (size_t colIndex = 0; colIndex < OMPResult[0].size(); ++colIndex) {
        maxError = std::max(std::abs(normalResult[rowIndex][colIndex] - OMPResult[rowIndex][colIndex]), maxError);
      }
    }
    std::cout << "Mayor error: " << std::to_string(maxError) << '\n';
    vt.addRow("OMP multiplication", ompTimeSeconds, maxError);

    // StrassenOMP multiplication
    std::vector<std::vector<float>> strassenOMPResult((1<<begin),std::vector<float>((1<<begin), 0.0));
    ti = omp_get_wtime();
    parallelStrassenMultiplication(matrix1,matrix2, strassenOMPResult, threshold);
    tf = omp_get_wtime();
    auto strassenOMPTimeSeconds = tf - ti;
    std::cout << "Tiempo multiplicación Strassen OMP: " << std::to_string(strassenOMPTimeSeconds) << '\n';
    maxError = 0.0;
    for (size_t rowIndex = 0; rowIndex < strassenOMPResult.size(); ++rowIndex) {
      for (size_t colIndex = 0; colIndex < strassenOMPResult[0].size(); ++colIndex) {
        maxError = std::max(std::abs(normalResult[rowIndex][colIndex] - strassenOMPResult[rowIndex][colIndex]), maxError);
      }
    }
    std::cout << "Mayor error: " << std::to_string(maxError) << '\n';
    vt.addRow("Strassen OMP multiplication", strassenOMPTimeSeconds, maxError);

    // -------------------------------------- CUDA --------------------------------------
    Matrix<float> CUDAMatrix1, CUDAMatrix2, CUDAResult;
    CUDAMatrix1.height = matrix1.size();
    CUDAMatrix1.width = matrix1[0].size();
    CUDAMatrix2.height = matrix2.size();
    CUDAMatrix2.width = matrix2[0].size();
    CUDAResult.height = matrix1.size();
    CUDAResult.width = matrix2[0].size();
    auto matrix1Flat = flatMatrix(matrix1);
    auto matrix2Flat = flatMatrix(matrix2);
    CUDAMatrix1.elements = matrix1Flat;
    CUDAMatrix2.elements = matrix2Flat;
    matrix1.clear();
    matrix2.clear();
    // CUDA simple multiplication
    ti = omp_get_wtime();
    multiplicationCudaWrapper(CUDAMatrix1, CUDAMatrix2, CUDAResult);
    tf = omp_get_wtime();
    auto CUDAMultTime = tf - ti;
    std::cout << "Tiempo multiplicación CUDA mult: " << std::to_string(CUDAMultTime) << '\n';
    maxError = 0.0;
    for (size_t rowIndex = 0; rowIndex < CUDAResult.height; ++rowIndex) {
      for (size_t colIndex = 0; colIndex < CUDAResult.width; ++colIndex) {
        maxError = std::max(std::abs(normalResult[rowIndex][colIndex] - CUDAResult.elements[rowIndex*CUDAResult.width + colIndex]), maxError);
      }
    }
    std::cout << "Mayor error: " << std::to_string(maxError) << '\n';
    vt.addRow("CUDA multiplication", CUDAMultTime, maxError);

    // CUDA strassen multiplication
    CUDAResult.elements.clear();
    ti = omp_get_wtime();
    CUDAStrassenMultiplication(CUDAMatrix1, CUDAMatrix2, CUDAResult, threshold);
    tf = omp_get_wtime();
    auto CUDAStrassenMult = tf - ti;
    std::cout << "Tiempo multiplicación CUDA Strassen mult: " << std::to_string(CUDAStrassenMult) << '\n';
    maxError = 0.0;
    for (size_t rowIndex = 0; rowIndex < CUDAResult.height; ++rowIndex) {
      for (size_t colIndex = 0; colIndex < CUDAResult.width; ++colIndex) {
        maxError = std::max(std::abs(normalResult[rowIndex][colIndex] - CUDAResult.elements[rowIndex*CUDAResult.width + colIndex]), maxError);
      }
    }
    std::cout << "Mayor error: " << std::to_string(maxError) << '\n';
    vt.addRow("CUDA Strassen multiplication", CUDAStrassenMult, maxError);
    vt.print(file);
  }
  file.close();
  return 0;
}
