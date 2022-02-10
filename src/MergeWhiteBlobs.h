//
// Created by Roman SAHEL on 09/02/2022.
//

#ifndef DEBOGGLER_MERGEWHITEBLOBS_H
#define DEBOGGLER_MERGEWHITEBLOBS_H

#include "commons.h"
#include "ProcessStep.h"
#include "FindWhiteBlobs.h"

struct MergeWhiteBlobs : ProcessStep {
    bool autoCompute = true;
    int size;
    int offset = 10;
    double epsilon;

    const char *GUILabel() override { return "Merge white blobs"; }

    void Process(const cv::Mat &src, cv::Mat &current) override {
        std::vector<std::vector<cv::Point> > contours;
        auto source = current.clone();

        if (autoCompute) {
            for (size = 10; size < 255; size += 10) {
                extractContours(current, contours, source);
                if (contours.size() == 1)
                    break;
            }
        } else {
            extractContours(current, contours, source);
        }

        if (contours.size() > 0) {
            current = src.clone();
            std::vector<std::vector<cv::Point> > hull(contours.size());
            convexHull(contours[0], hull[0]);

            auto peri = cv::arcLength(hull[0], true);
            std::vector<cv::Point> simplifiedHull = hull[0];
            if (autoCompute) {
                for (epsilon = 0.12; epsilon < 10.0; epsilon += 0.01) {
                    simplifyHull(peri, hull[0], simplifiedHull);
                    if (simplifiedHull.size() <= 4)
                        break;
                }
            } else {
                simplifyHull(peri, hull[0], simplifiedHull);
            }
            hull[0] = simplifiedHull;
            fourPointTransform(src, current, simplifiedHull);
        }
    }

    void extractContours(cv::Mat &current, std::vector<std::vector<cv::Point>> &contours, const cv::Mat &source) const {
        auto element = getStructuringElement(0, cv::Size(2 * size + 1, 2 * size + 1), cv::Point(size, size));
        dilate(source, current, element);
        findContours(current, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    }

    void simplifyHull(double peri, const std::vector<cv::Point> &baseHull, std::vector<cv::Point> &simplifiedHull) const {
        cv::approxPolyDP(baseHull, simplifiedHull, epsilon * 0.1 * peri, true);
    }

    void fourPointTransform(const cv::Mat &src, cv::Mat &current, const std::vector<cv::Point> &points) {
        cv::Point2f topLeft;
        cv::Point2f topRight;
        cv::Point2f bottomRight;
        cv::Point2f bottomLeft;

        int smallestSum = std::numeric_limits<int>::max(), largestSum = std::numeric_limits<int>::min();
        int smallestDiff = std::numeric_limits<int>::max(), largestDiff = std::numeric_limits<int>::min();
        for (int i = 0; i < points.size(); ++i) {
            int sum = points[i].x + points[i].y;
            int diff = points[i].x - points[i].y;
            if (sum < smallestSum) {
                smallestSum = sum;
                topLeft = points[i];
            }
            if (sum > largestSum) {
                largestSum = sum;
                bottomRight = points[i];
            }
            if (diff < smallestDiff) {
                smallestDiff = diff;
                bottomLeft = points[i];
            }
            if (diff > largestDiff) {
                largestDiff = diff;
                topRight = points[i];
            }
        }

        auto topWidth = topRight.x - topLeft.x;
        auto bottomWidth = bottomRight.x - bottomLeft.x;
        auto rightHeight = bottomRight.y - topRight.y;
        auto leftHeight = bottomLeft.y - topLeft.y;

        float xOffset = offset, yOffset = offset;

        current = cv::Mat::zeros(std::max(leftHeight, rightHeight) + 2 * yOffset, std::max(topWidth, bottomWidth) + 2 * xOffset, src.type());
        cv::Point2f orderedPoints[] = {
                topLeft + cv::Point2f(-xOffset, -yOffset),
                topRight + cv::Point2f(xOffset, -yOffset),
                bottomRight + cv::Point2f(xOffset, yOffset),
                bottomLeft + cv::Point2f(-xOffset, yOffset),
        };
        cv::Point2f straightPoints[] = {
                cv::Point2f(0, 0),
                cv::Point2f(current.cols, 0),
                cv::Point2f(current.cols, current.rows),
                cv::Point2f(0, current.rows),
        };
//        cv::circle(current, topLeft, 32, cv::Scalar(255, 255, 255));
//        cv::circle(current, topRight, 24, cv::Scalar(255, 255, 255));
//        cv::circle(current, bottomRight, 18, cv::Scalar(255, 255, 255));
//        cv::circle(current, bottomLeft, 12, cv::Scalar(255, 255, 255));

        auto transform = cv::getPerspectiveTransform(orderedPoints, straightPoints);
        cv::warpPerspective(src, current, transform, current.size());
    }

    bool DrawGUI(const cv::Rect &window) override {
        bool hasChanged = false;
        hasChanged |= cvui::checkbox("Auto compute", &autoCompute);
        hasChanged |= trackbar("Size", window, size, 0, 255);
        hasChanged |= trackbar("Epsilon", window, epsilon, 0.0, 10.0);
        hasChanged |= trackbar("Offset", window, offset, 0, 100);
        return hasChanged;
    }

};

#endif //DEBOGGLER_MERGEWHITEBLOBS_H
