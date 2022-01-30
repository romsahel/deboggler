//
// Created by Roman SAHEL on 29/01/2022.
//

#ifndef DEBOGGLER_FINDWHITEBLOBS_H
#define DEBOGGLER_FINDWHITEBLOBS_H

#include "commons.h"
#include "ProcessStep.h"

struct FindWhiteBlobs : ProcessStep {
    bool autoSensitivity = true;
    int sensitivity = 75;
    bool test = false;
    std::vector<std::vector<cv::Point> > contours;

    FindWhiteBlobs(bool t = false) : test(t) {}
    
    const char *GUILabel() override { return "Floodfill"; }

    void Process(const cv::Mat &src, cv::Mat &current) override {
        contours.clear();
        if (current.channels() == 1)
            cv::cvtColor(current, current, cv::COLOR_GRAY2BGR);    
        cv::cvtColor(current, current, cv::COLOR_BGR2HSV);
        cv::Mat source = current.clone();
        cv::Rect safeArea(src.cols * 0.05f, src.rows * 0.05f, src.cols - src.cols * 0.05f, src.rows - src.rows * 0.05f);
        int idealArea = src.cols * src.rows / 4;
        int maxArea = idealArea * 1.5f;
        int minArea = idealArea * 0.01f;
        if (autoSensitivity) {
            for (sensitivity = 127; sensitivity >= 0; sensitivity--) {
                extractContours(current, source, contours, safeArea, minArea, maxArea);
                if (contours.size() == 16) { 
                    break; 
                }
            }
        } else {
            extractContours(current, source, contours, safeArea, minArea, maxArea);
        }

        if (contours.size() != 16 || !test) {
            cv::Mat result = cv::Mat::zeros(current.size(), current.type());
            for (size_t i = 0; i < contours.size(); i++) {
//            auto rect = minAreaRect(contours[i]);
//            printf("area: %f // %fx%f=%f\n", rect.size.area(), rect.size.width, rect.size.height, std::abs(1.0f - rect.size.width / rect.size.height));
                drawContours(result, contours, (int) i, cv::Scalar(255, 255, 255), -1);
            }

            current = result;
        }
    }

    template<typename T>
    static bool approximately(T a, T b, T epsilon) {
        return std::abs(b - a) < epsilon;
    }

    void extractContours(cv::Mat &current, cv::Mat &source, std::vector<std::vector<cv::Point>> &contours, const cv::Rect& safeArea, int minArea, int maxArea) const {
        cv::inRange(source, cv::Vec3b(0, 0, 255 - sensitivity), cv::Vec3b(255, sensitivity, 255), current);
        cv::threshold(current, current, 127, 255, 0);
        findContours(current, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        contours.erase(
                std::remove_if(contours.begin(), contours.end(),
                               [&safeArea, minArea, maxArea, test=this->test](const auto &o) {
                                   cv::RotatedRect rect = minAreaRect(o);
                                   bool remove = rect.size.area() < minArea || rect.size.area() > maxArea || (!approximately(rect.size.width / rect.size.height, 1.0f, 0.2f));
                                   remove = remove || (!test && !safeArea.contains(rect.center));
//                                   printf("area: %f // %fx%f=%f\n", rect.size.area(), rect.size.width, rect.size.height, std::abs(1.0f - rect.size.width / rect.size.height));
                                   return remove;
                               }),
                contours.end());

    }

    bool DrawGUI(const cv::Rect &window) override {
        bool hasChanged = false;
        cvui::checkbox("Auto-sensitivity", &autoSensitivity);
        hasChanged |= trackbar("Sensitivity", window, sensitivity, 0, 255);
        return hasChanged;
    }

};


#endif //DEBOGGLER_FINDWHITEBLOBS_H
