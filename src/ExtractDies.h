//
// Created by Roman SAHEL on 09/02/2022.
//

#ifndef DEBOGGLER_EXTRACTDIES_H
#define DEBOGGLER_EXTRACTDIES_H

#include "commons.h"
#include "ProcessStep.h"
#include "Assembly.h"
#include "FindWhiteBlobs.h"

struct ExtractDies : ProcessStep {
    int denoisingStrength = 1;
    const Assembly &assembly;
    std::vector<ProcessStep *> substeps = {
            new FindWhiteBlobs(true),
    };

    ExtractDies(const Assembly &assembly) : assembly(assembly) {}

    const char *GUILabel() override { return "Box"; }

    void Process(const cv::Mat &src, cv::Mat &current) override {
        auto copy = current.clone();
        for (int i = 0; i < substeps.size(); ++i) {
            substeps[i]->Process(src, current);
        }

        cv::Mat kernelOne = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(denoisingStrength * 2 + 1, denoisingStrength * 2 + 1));
        cv::morphologyEx(current, current, cv::MORPH_CLOSE, kernelOne);

        std::vector<std::vector<cv::Point> > &contours = ((FindWhiteBlobs *) substeps[0])->contours;
        if (contours.size() != 16)
            return;

        std::vector<std::vector<cv::Point> > contours_poly(contours.size());
        std::vector<cv::Rect> boundRect(contours.size());
        for (size_t i = 0; i < contours.size(); i++) {
            approxPolyDP(contours[i], contours_poly[i], 3, true);
            boundRect[i] = boundingRect(contours_poly[i]);
        }


        auto white = cv::Scalar(255, 255, 255);
        int maxWidth = 28, maxHeight = 28;

        std::sort(std::begin(boundRect), std::end(boundRect), [](auto &a, auto &b) {
            return a.y < b.y;
        });
        for (int i = 0; i < 4; ++i) {
            std::sort(std::begin(boundRect) + i * 4, std::begin(boundRect) + (i + 1) * 4, [](auto &a, auto &b) {
                return a.x < b.x;
            });
        }

        if (false) {
            for (int i = 0; i < boundRect.size(); ++i) {
                cv::rectangle(current, boundRect[i], white);
            }

            return;
        }

        copy = cv::Mat(cv::Size(maxWidth * 8, maxHeight * 8), current.type(), white);

        int copyX = 0, copyY = 0;
        auto folder = "output/" + assembly.targets[assembly.sourceIndex] + '/';
        if (!std::filesystem::is_directory(folder) || !std::filesystem::exists(folder)) { // Check if src folder exists
            std::filesystem::create_directory(folder); // create src folder
        }
        for (size_t i = 0; i < boundRect.size(); i++) {
            cv::Mat mat = current(boundRect[i]);
            auto rect = findLastContours(mat);

            boundRect[i].x += rect.x;
            boundRect[i].y += rect.y;
            boundRect[i].width = rect.width;
            boundRect[i].height = rect.height;

            mat = current(boundRect[i]);
            if (mat.cols > mat.rows) {
                cv::Mat resize = cv::Mat(mat.cols, mat.cols, mat.type(), 255);
                mat.copyTo(resize(cv::Rect(0, (mat.cols - mat.rows) / 2, mat.cols, mat.rows)));
                mat = resize;
            } else {
                cv::Mat resize = cv::Mat(mat.rows, mat.rows, mat.type(), 255);
                mat.copyTo(resize(cv::Rect((mat.rows - mat.cols) / 2, 0, mat.cols, mat.rows)));
                mat = resize;
            }
            cv::resize(mat, mat, cv::Size(maxWidth, maxHeight), cv::INTER_AREA);

            mat.copyTo(copy(cv::Rect(copyX, copyY, mat.cols, mat.rows)));
            if (i == 3 || i == 7 || i == 11) {
                copyX = 0;
                copyY += mat.rows;
            } else {
                copyX += mat.cols;
            }

            int img_index = 0;
            for (int j = 0; j < 4; ++j, img_index++) {
                auto filename = folder + std::to_string(i) + '_' + std::to_string(img_index) + '_' + assembly.targets[assembly.sourceIndex][i] + ".jpg";
                imwrite(filename, mat);
                cv::rotate(mat, mat, cv::ROTATE_90_CLOCKWISE);
            }
            addSaltAndPepper(mat);
            for (int j = 0; j < 4; ++j, img_index++) {
                auto filename = folder + std::to_string(i) + '_' + std::to_string(img_index) + '_' + assembly.targets[assembly.sourceIndex][i] + ".jpg";
                cv::rotate(mat, mat, cv::ROTATE_90_CLOCKWISE);
            }
        }

        current = copy;
    }

    void addSaltAndPepper(cv::Mat& srcArr, float pa = 0.05, float pb = 0.05)
    {
        cv::RNG rng;
        int amount1 = srcArr.rows * srcArr.cols * pa;
        int amount2 = srcArr.rows * srcArr.cols * pb;
        for (int counter = 0; counter < amount1; ++counter) {
            srcArr.at<uchar>(rng.uniform(0, srcArr.rows), rng.uniform(0, srcArr.cols)) = 0;

        }
        for (int counter = 0; counter < amount2; ++counter) {
            srcArr.at<uchar>(rng.uniform(0, srcArr.rows), rng.uniform(0, srcArr.cols)) = 255;
        }
    }
    
        cv::Rect findLastContours(cv::Mat &mat) {
            std::vector<std::vector<cv::Point> > contours;
            findContours(mat, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    
            std::vector<std::vector<cv::Point> > contours_poly(contours.size());
            std::vector<cv::Rect> boundRect(contours.size());
            for (size_t i = 0; i < contours.size(); i++) {
                approxPolyDP(contours[i], contours_poly[i], 3, true);
                boundRect[i] = boundingRect(contours_poly[i]);
            }
            std::sort(std::begin(boundRect), std::end(boundRect), [](auto &a, auto &b) {
                return a.size().area() > b.size().area();
            });
            return boundRect.size() > 1 ? boundRect[1] : boundRect[0];
        }

    bool DrawGUI(const cv::Rect &window) override {
        bool hasChanged = false;
        for (int i = 0; i < substeps.size(); ++i) {
            hasChanged |= substeps[i]->DrawGUI(window);
        }
        hasChanged |= trackbar("Denoising strength", window, denoisingStrength, 0, 10);
        return hasChanged;
    }

};

#endif //DEBOGGLER_EXTRACTDIES_H
