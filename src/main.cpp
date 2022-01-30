#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/photo.hpp>

#include "commons.h"
#include "ProcessStep.h"
#include "FindWhiteBlobs.h"
#include "Assembly.h"

struct ExtractDies : ProcessStep {
    int denoisingStrength = 1;
    std::vector<ProcessStep *> substeps = {
            new FindWhiteBlobs(true),
    };

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
        int maxWidth = 0, maxHeight = 0;
        for (size_t i = 0; i < std::min((int) boundRect.size(), 16); i++) {
            if (boundRect[i].size().width > maxWidth)
                maxWidth = boundRect[i].size().width;
            if (boundRect[i].size().height > maxHeight)
                maxHeight = boundRect[i].size().height;
        }
        
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

        copy = cv::Mat(cv::Size(maxWidth * 4, maxHeight * 4), current.type(), white);

        int copyX = 0, copyY = 0;
        auto filePrefix = "output/" + std::string(filename) + '_';
        for (size_t i = 0; i < boundRect.size(); i++) {
            cv::Mat mat = current(boundRect[i]);
            auto lowDiff = cv::Vec3b(127, 127, 127);
            auto highDiff = cv::Vec3b(127, 127, 127);
            for (int x = 0; x < mat.cols; ++x) {
                cv::floodFill(mat, cv::Point(x, 0), white, nullptr, lowDiff, highDiff);
                cv::floodFill(mat, cv::Point(x, mat.rows - 1), white, nullptr, lowDiff, highDiff);
            }
            for (int y = 1; y < mat.rows - 1; ++y) {
                cv::floodFill(mat, cv::Point(0, y), white, nullptr, lowDiff, highDiff);
                cv::floodFill(mat, cv::Point(mat.cols - 1, y), white, nullptr, lowDiff, highDiff);
            }

            mat.copyTo(copy(cv::Rect(copyX, copyY, boundRect[i].width, boundRect[i].height)));
            if (i == 3 || i == 7 || i == 11) {
                copyX = 0;
                copyY += maxHeight;
            } else {
                copyX += maxWidth;
            }


            imwrite(filePrefix + std::to_string(i) + ".jpg", mat);
        }

        current = copy;
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

int main(int argc, const char *argv[]) {

    Assembly assembly;
    assembly.showMask = false;
    assembly.push_back(new FindWhiteBlobs());
    assembly.push_back(new MergeWhiteBlobs());
    assembly.push_back(new ExtractDies());
    assembly.init();

    while (true) {

//        if (cvui::button(uiFrame, uiFrame.cols - 50, uiFrame.rows - 30, "&Quit")) {
//            break;
//        }

        assembly.update();
        assembly.draw();

        cvui::update();

        assembly.show();
    }

    return 0;
}