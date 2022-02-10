//
// Created by Roman SAHEL on 09/02/2022.
//

#ifndef DEBOGGLER_SERIALIZATION_H
#define DEBOGGLER_SERIALIZATION_H

#include <opencv2/core/core.hpp>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

void matwrite(std::ofstream &fs, const Mat &mat) {
    // Header
    int type = mat.type();
    int channels = mat.channels();
    fs.write((char *) &mat.rows, sizeof(int));    // rows
    fs.write((char *) &mat.cols, sizeof(int));    // cols
    fs.write((char *) &type, sizeof(int));        // type
    fs.write((char *) &channels, sizeof(int));    // channels

    // Data
    if (mat.isContinuous()) {
        fs.write(mat.ptr<char>(0), (mat.dataend - mat.datastart));
    } else {
        int rowsz = CV_ELEM_SIZE(type) * mat.cols;
        for (int r = 0; r < mat.rows; ++r) {
            fs.write(mat.ptr<char>(r), rowsz);
        }
    }
}

Mat matread(ifstream &fs) {
    // Header
    int rows, cols, type, channels;
    fs.read((char *) &rows, sizeof(int));         // rows
    fs.read((char *) &cols, sizeof(int));         // cols
    fs.read((char *) &type, sizeof(int));         // type
    fs.read((char *) &channels, sizeof(int));     // channels

    // Data
    Mat mat(rows, cols, type);
    fs.read((char *) mat.data, CV_ELEM_SIZE(type) * rows * cols);

    return mat;
}

#endif //DEBOGGLER_SERIALIZATION_H
