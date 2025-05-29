#ifndef MATRIX_BASIC_HPP
#define MATRIX_BASIC_HPP
#include <vector>
#include <cmath>
#include <stdexcept>
#include <fstream>
#include <string>
#include <iostream>
class Matrix {
private:
    size_t rows;
    size_t cols;
    std::vector<float> data;
public:
    // 构造函数
    Matrix() : rows(0), cols(0) {}
    Matrix(size_t r, size_t c) : rows(r), cols(c), data(r* c, 0.0f) {}
    Matrix(const std::vector<float>& vec, bool isRowVector = true) {
        if (isRowVector) {
            rows = 1;
            cols = vec.size();
        }else {
            rows = vec.size();
            cols = 1;}
        data = vec;}
    size_t getRows() const { return rows; }
    size_t getCols() const { return cols; }
    float& operator()(size_t i, size_t j) {
        return data[i * cols + j];}

    const float& operator()(size_t i, size_t j) const {
        return data[i * cols + j];}

    Matrix operator+(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions must match for addition.");}
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows * cols; ++i) {
            result.data[i] = data[i] + other.data[i];}
        return result;}

    Matrix operator*(const Matrix& other) const {
        if (cols != other.rows) {
            throw std::invalid_argument("Number of columns in the first matrix must be equal to the number of rows in the second matrix.");  }

        Matrix result(rows, other.cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < other.cols; ++j) {
                for (size_t k = 0; k < cols; ++k) {
                    result(i, j) += (*this)(i, k) * other(k, j);}}}
        return result;}

    bool loadFromBinaryFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return false;}
        file.seekg(0, std::ios::end);
        size_t fileSize = file.tellg();
        file.seekg(0, std::ios::beg);
        if (fileSize != rows * cols * sizeof(float)) {
            std::cerr << "File size does not match matrix dimensions: " << filename << std::endl;
            return false;}
        data.resize(rows * cols);
        file.read(reinterpret_cast<char*>(data.data()), fileSize);
        return true;
    }
    void print() const {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                std::cout << (*this)(i, j) << " ";}
            std::cout << std::endl;}}};

Matrix relu(const Matrix& matrix) {
    Matrix result(matrix.getRows(), matrix.getCols());
    for (size_t i = 0; i < matrix.getRows(); ++i) {
        for (size_t j = 0; j < matrix.getCols(); ++j) {
            result(i, j) = std::max(0.0f, matrix(i, j));
        }
    }
    return result;
}

Matrix softmax(const Matrix& matrix) {
    if (matrix.getRows() != 1 && matrix.getCols() != 1) {
        throw std::invalid_argument("Softmax input a vector.");
    }
    Matrix result = matrix;
    float maxVal = -INFINITY;
    for (size_t i = 0; i < matrix.getRows(); ++i) {
        for (size_t j = 0; j < matrix.getCols(); ++j) {
            maxVal = std::max(maxVal, matrix(i, j));}}
    float sum = 0.0;
    for (size_t i = 0; i < result.getRows(); ++i) {
        for (size_t j = 0; j < result.getCols(); ++j) {
            result(i, j) = std::exp(matrix(i, j) - maxVal);
            sum += result(i, j);
        }
    }
    for (size_t i = 0; i < result.getRows(); ++i) {
        for (size_t j = 0; j < result.getCols(); ++j) {
            result(i, j) /= sum;}
    }
    return result;
}

class Model {
private:
    Matrix weight1; 
    Matrix bias1;   
    Matrix weight2; 
    Matrix bias2;    
public:
    Model(const Matrix& w1, const Matrix& b1,
        const Matrix& w2, const Matrix& b2)
        : weight1(w1), bias1(b1), weight2(w2), bias2(b2) {}
    Matrix forward(const Matrix& input) {
        Matrix temp = input * weight1;
        temp = temp + bias1;
        temp = relu(temp);
        temp = temp * weight2;
        temp = temp + bias2;
        temp = softmax(temp);
        return temp;
    }
};
#endif 
