#pragma once

#include <opencv2/imgcodecs.hpp>
#include "/Library/dev/rsahel/deboggler-repo/src/neuralnetwork/neuralnetwork.h"

enum class ProcessResult {
    DicesNotFound,
    BlobsNotMerged,
    FrameNotFound,
    IndividualDicesNotFound,
    PROCESS_FAILURE,
    BoardIsolated,
    DicesFound,
    BlobsMerged,
    FrameFound,
    CornersFound,
    Warped,
    WarpedAndIsolated,
    WarpedAndIsolatedAndCleaned,
    IndividualDicesFound,
    IndividualDicesFoundAndMerged,
    PROCESS_SUCCESS,
    PROCESS_SUCCESS_INDECISIVE,
};

#define CHECK_MAX_STEP(CURRENTSTEP, MAXSTEP, ...) do {\
        if (int(CURRENTSTEP) >= int(MAXSTEP)) {        \
            __VA_ARGS__;                              \
            return CURRENTSTEP;                       \
        }                                             \
    } while (false)                                   \


struct Deboggler {
    int low_s = 255 - 125;
    int high_h = 25 + 125;
    int canny_threshold = 180;
    
    
    ProcessResult maxStep = ProcessResult::PROCESS_SUCCESS;
#ifdef WRITE_IMAGE
    std::string imageName;
#endif

    std::vector<std::vector<cv::Point>> contours;
    std::vector<std::vector<cv::Point>> hull{1};
    std::vector<cv::Point> simplifiedHull;
    std::vector<cv::RotatedRect> diceRotatedRects;
    cv::Point2f orderedPoints[4];
    cv::Point2f straightPoints[4];

    cv::Mat transformed;
    uint16_t *result;

    NeuralNetwork neuralNetwork;

    void (*foo)(const char *);

    void isolateBoggleBoard(cv::Mat& src, cv::Mat &mask, int cannyThreshold1, cv::Rect *roi = nullptr) {
        cv::cvtColor(src, mask, cv::COLOR_BGR2HSV);
        cv::Vec3b lower(0, low_s, 0);
        cv::Vec3b upper(high_h, 255, 255);
        cv::inRange(mask, lower, upper, mask);
        cv::bitwise_not(mask, mask);
        src.copyTo(mask, mask);
        if (roi != nullptr)
            mask = mask(*roi);
        
        static cv::Mat canny;
        if (cannyThreshold1 < 255) {
            cv::Canny(mask, canny, cannyThreshold1, 255);
            auto element = getStructuringElement(0, cv::Size(3, 3));
            dilate(canny, canny, element);
            cv::bitwise_not(canny, canny);
        }

        cv::cvtColor(mask, mask, cv::COLOR_BGR2GRAY);
        cv::threshold(mask, mask, 127, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        if (cannyThreshold1 < 255) {
            cv::bitwise_and(mask, canny, mask);
        }
    }

    static int findDicesContours(cv::Mat &mask, cv::Rect &roi,
                                  std::vector<std::vector<cv::Point>> &contours,
                                  std::vector<cv::Point> &hull) {
        const auto roiSize = roi.size();
        findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        mask = 0;
        int count = 0;
        for (int i = 0; i < contours.size(); ++i) {
            auto rect = minAreaRect(contours[i]);
            cv::convexHull(contours[i], hull);
            bool isInvalid = false;
            isInvalid = isInvalid || rect.size.width > roiSize.width / 3;
            isInvalid = isInvalid || rect.size.height > roiSize.height / 3;
            isInvalid = isInvalid || rect.size.width < roiSize.width / 8;
            isInvalid = isInvalid || rect.size.height < roiSize.height / 8;
            isInvalid = isInvalid || std::abs(1.0f - float(rect.size.aspectRatio())) > 0.2f;
            auto solidity = (cv::contourArea(contours[i]) / cv::contourArea(hull));
            isInvalid = isInvalid || std::abs(1.0f - solidity) > 0.4f;
            if (!isInvalid) {
                drawContours(mask, contours, (int) i, count++ < 16 ? 255 : 0, -1);
            }
        }
        return count;
    }

    static cv::RotatedRect mergeRotatedRect(cv::RotatedRect &r1, cv::RotatedRect &r2) {
        static vector<Point2f> allpts;
        allpts.clear();

        static Point2f p1[4];
        r1.points(p1);
        allpts.push_back(p1[0]);
        allpts.push_back(p1[1]);
        allpts.push_back(p1[2]);
        allpts.push_back(p1[3]);
        r2.points(p1);
        allpts.push_back(p1[0]);
        allpts.push_back(p1[1]);
        allpts.push_back(p1[2]);
        allpts.push_back(p1[3]);

        auto item = minAreaRect(allpts);
        return item;
    }

    static void validateROIFor(cv::Rect &rect, const cv::Mat &src) {
        if (rect.x < 0) rect.x = 0;
        if (rect.y < 0) rect.y = 0;
        if (rect.x + rect.width >= src.cols) rect.width = src.cols - rect.x - 1;
        if (rect.y + rect.height >= src.rows) rect.height = src.rows - rect.y - 1;
    }

    static void drawFrameAndCorners(cv::Mat &src, cv::Mat &mask, const cv::Rect& srcRoi,
                                    std::vector<std::vector<cv::Point>> &hulls, std::vector<cv::Point> &simplifiedHull) {
        mask = 0;
        drawContours(mask, hulls, 0, 255, 15);
        for (int i = 0; i < 4; ++i) {
            cv::circle(mask, simplifiedHull[i], 16 + 4 * i, cv::Scalar(255, 255, 255), -1);
        }
        cv::cvtColor(mask, mask, cv::COLOR_GRAY2BGR);
        cv::bitwise_or(src(srcRoi), mask, mask);
    }

    static void drawCorners(cv::Mat &src, cv::Mat &mask, cv::Point2f *orderedPoints) {
        mask = 0;
        for (int i = 0; i < 4; ++i) {
            cv::circle(mask, orderedPoints[i], 16 + 4 * i, cv::Scalar(255, 255, 255), -1);
        }
    }

    static void cleanIsolatedDices(cv::Mat &mask) {
        cv::floodFill(mask, cv::Point(mask.size().width / 2, mask.size().height / 2), 255);
        cv::floodFill(mask, cv::Point(0, 0), 255);
        cv::floodFill(mask, cv::Point(mask.size().width - 1, 0), 255);
        cv::floodFill(mask, cv::Point(mask.size().width - 1, mask.size().height - 1), 255);
        cv::floodFill(mask, cv::Point(0, mask.size().height - 1), 255);
        cv::bitwise_not(mask, mask);
    }

    static void findOrderedIndivididualDicesContours(cv::Mat &mask,
                                                     std::vector<std::vector<cv::Point>> &contours,
                                                     std::vector<RotatedRect> &diceRotatedRects) {
        diceRotatedRects.clear();
        findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        for (int i = 0; i < contours.size(); ++i) {
            auto rect = minAreaRect(contours[i]);
            bool isInvalid = false;
            isInvalid = isInvalid || rect.size.area() < 300;
            isInvalid = isInvalid || rect.size.aspectRatio() > 6;
            if (!isInvalid) {
                diceRotatedRects.push_back(rect);
            }
        }

        // Sort vertically
        std::sort(std::begin(diceRotatedRects), std::end(diceRotatedRects), [](auto &a, auto &b) {
            return a.center.y < b.center.y;
        });

        // Sort each line horizontally
        auto lineHeight = mask.rows * 0.25f;
        auto lineStart = std::begin(diceRotatedRects);
        for (int i = 0; i < 4; ++i) {
            auto lineEnd = lineStart + 1;
            while (lineEnd != std::end(diceRotatedRects)
                   && lineEnd->center.y < float(i + 1) * lineHeight) {
                lineEnd++;
            }
            std::sort(lineStart, lineEnd, [](auto &a, auto &b) {
                return a.center.x < b.center.x;
            });
            lineStart = lineEnd;
        }
    }

    static void mergeRelatedContours(cv::Mat &mask, std::vector<RotatedRect> &diceRotatedRects) {
        std::vector<int> indicesToRemove;
        auto lineHalfHeight = mask.rows / 8.0f;
        auto columnHalfWidth = mask.cols / 8.0f;
        for (int i = 0; i < diceRotatedRects.size(); ++i) {
            auto scaledRect = diceRotatedRects[i];
            scaledRect.size *= 2.0f;
            static cv::Point2f corners[4];
            if (i + 1 < diceRotatedRects.size()) {
                bool shouldMerge = false;
//                diceRotatedRects[i + 1].points(corners);
//                for (int k = 0; k < 4 && !shouldMerge; ++k) {
//                    shouldMerge = scaledRect.boundingRect2f().contains(corners[k]);
//                }
                shouldMerge = (diceRotatedRects[i + 1].center.x - diceRotatedRects[i].center.x) < columnHalfWidth
                              && (diceRotatedRects[i + 1].center.y - diceRotatedRects[i].center.y) < lineHalfHeight;
                if (shouldMerge) {
                    diceRotatedRects[i] = mergeRotatedRect(diceRotatedRects[i + 1], diceRotatedRects[i]);
                    indicesToRemove.push_back(i + 1);
                    i++;
                }
            }
        }

        if (!indicesToRemove.empty()) {
            for (int i = int(indicesToRemove.size()) - 1; i >= 0; --i) {
                auto it = diceRotatedRects.begin();
                std::advance(it, indicesToRemove[i]);
                diceRotatedRects.erase(it);
            }
        }
    }

    ProcessResult Process(cv::Mat &src, cv::Mat &mask) {
        contours.clear();

        int size;
        int x = 0, y = 0;
        if (src.size().height >= src.size().width) {
            size = src.size().width;
            y = (src.size().height - src.size().width) / 2;
        } else {
            size = src.size().height;
            x = (src.size().width - src.size().height) / 2;
        }

        auto roi = cv::Rect(x, y, size, size);
        isolateBoggleBoard(src, mask, canny_threshold, &roi);
        cv::floodFill(mask, cv::Point(0, 0), 0);
        cv::floodFill(mask, cv::Point(mask.size().width - 1, 0), 0);
        cv::floodFill(mask, cv::Point(mask.size().width - 1, mask.size().height - 1), 0);
        cv::floodFill(mask, cv::Point(0, mask.size().height - 1), 0);
//        for (int i = 0; i < mask.size().width; i += 2) {
//            cv::floodFill(mask, cv::Point(i, 0), 0);
//            cv::floodFill(mask, cv::Point(i, mask.size().height - 1), 0);
//        }
//        for (int i = 0; i < mask.size().height; i += 2) {
//            cv::floodFill(mask, cv::Point(0, i), 0);
//            cv::floodFill(mask, cv::Point(mask.size().width - 1, i), 0);
//        }
        CHECK_MAX_STEP(ProcessResult::BoardIsolated, maxStep);

        int diceCount = findDicesContours(mask, roi, contours, simplifiedHull);
        if (diceCount < 16)
            return ProcessResult::DicesNotFound;
        CHECK_MAX_STEP(ProcessResult::DicesFound, maxStep);


        if (!mergeBlobs(mask, contours))
            return ProcessResult::BlobsNotMerged;
        CHECK_MAX_STEP(ProcessResult::BlobsMerged, maxStep);

        if (!findFrameFromContours(0, contours, hull, simplifiedHull))
            return ProcessResult::FrameNotFound;

        CHECK_MAX_STEP(ProcessResult::FrameFound, maxStep, drawFrameAndCorners(src, mask, roi, hull, simplifiedHull));

        auto cornerRect = orderAndSetCorners(src, simplifiedHull, orderedPoints, straightPoints);
        CHECK_MAX_STEP(ProcessResult::CornersFound, maxStep, drawCorners(src, mask, orderedPoints));

        auto transform = cv::getPerspectiveTransform(orderedPoints, straightPoints);
        src(roi).copyTo(transformed, mask);
        cv::warpPerspective(transformed, transformed, transform, cornerRect.size());
        CHECK_MAX_STEP(ProcessResult::Warped, maxStep);

        isolateBoggleBoard(transformed, mask, 255);
        CHECK_MAX_STEP(ProcessResult::WarpedAndIsolated, maxStep);

        cleanIsolatedDices(mask);
        CHECK_MAX_STEP(ProcessResult::WarpedAndIsolatedAndCleaned, maxStep);


        findOrderedIndivididualDicesContours(mask, contours, diceRotatedRects);
        if (diceRotatedRects.size() < 16)
            return ProcessResult::IndividualDicesNotFound;
        CHECK_MAX_STEP(ProcessResult::IndividualDicesFound, maxStep);

        mergeRelatedContours(mask, diceRotatedRects);
        CHECK_MAX_STEP(ProcessResult::IndividualDicesFoundAndMerged, maxStep);
        if (diceRotatedRects.size() != 16)
            return ProcessResult::IndividualDicesNotFound;

#ifdef WRITE_IMAGE
        drawLinedupSortedRects(mask, transformed, diceRotatedRects);
#endif

#ifdef WRITE_IMAGE
        auto folder = std::filesystem::path("../output/");
        if (!imageName.empty()) {
            if (!std::filesystem::is_directory(folder) || !std::filesystem::exists(folder)) { // Check if src folder exists
                std::filesystem::create_directory(folder); // create src folder
            }
        }
#endif
        auto background = cv::Scalar(0);
        int maxSize = 28;
        float averageScore = 0;
#ifdef WRITE_IMAGE
        transformed = cv::Mat(cv::Size(maxSize * 4, maxSize * 4), CV_8UC1, background);
        cv::Rect dstRoi(0, 0, maxSize, maxSize);
#endif
        for (int i = 0; i < diceRotatedRects.size() && i < 16; ++i) {
            cv::Mat mat;
            extractAndStraighten(mask, mat, diceRotatedRects[i]);
            resizeAndFitACenter(mat, cv::Size(maxSize, maxSize), background);

#ifdef WRITE_IMAGE
            writeCharacterToFile(mat, transformed, &dstRoi, i, imageName, folder, false);
#endif
#ifdef FEEDFORWARD
            mat = mat.reshape(1, mat.cols * mat.rows);
            mat.convertTo(mat, CV_32FC1, 1.0f / 255.0f);
            auto guess = neuralNetwork.feed_forward(mat);
            int maxIndex = 0;
            for (int k = 1; k < 26; ++k) {
                if (guess.at<float>(0, k) > guess.at<float>(0, maxIndex)) {
                    maxIndex = k;
                }
            }
            char guessedChar = (char) ('A' + maxIndex);
            averageScore += guess.at<float>(0, maxIndex);
//            log("%c (%f)\n", guessedChar, guess.at<float>(0, maxIndex));
            result[i] = guessedChar;
#endif
        }

        averageScore /= 16.0f;
        return averageScore > 0.97f ? ProcessResult::PROCESS_SUCCESS : ProcessResult::PROCESS_SUCCESS_INDECISIVE;
    }

    static void writeCharacterToFile(cv::Mat &src, cv::Mat &dst, cv::Rect *dstRoi, int index,
                                     const std::string &imageName,
                                     const std::filesystem::path &folder,
                                     bool useSaltAndPepper) {
        if (dstRoi != nullptr) {
            src.copyTo(dst(*dstRoi));
            if (index == 3 || index == 7 || index == 11) {
                dstRoi->x = 0;
                dstRoi->y += src.rows;
            } else {
                dstRoi->x += src.cols;
            }
        }
        if (!imageName.empty()) {
            int img_index = 0;
            for (int n = 0; n < (useSaltAndPepper ? 2 : 1); ++n) {
                for (int j = 0; j < 4; ++j, img_index++) {
                    auto filename = std::string();
                    filename.push_back(imageName[index]);
                    filename.push_back('_');
                    filename += imageName;
                    filename.push_back('_');
                    filename += std::to_string(img_index);
                    filename += ".jpg";
                    cv::imwrite((folder / filename).string(), src);
                    cv::rotate(src, src, cv::ROTATE_90_CLOCKWISE);
                }
                addSaltAndPepper(src);
            }
        }
    }

    static void drawLinedupSortedRects(cv::Mat &mask, cv::Mat &dst, const std::vector<RotatedRect> &rects) {
        int totalWidth = 0;
        int maximumHeight = 0;
        for (int i = 0; i < rects.size(); ++i) {
            auto &rect = rects[i];
            totalWidth += rect.boundingRect().size().width;
            if (maximumHeight < rect.boundingRect().size().height)
                maximumHeight = rect.boundingRect().size().height;
        }

        int xDstOffset = 0;
        dst = cv::Mat::zeros(maximumHeight, totalWidth, CV_8UC1);
        for (int i = 0; i < rects.size(); ++i) {
            Rect rect = rects[i].boundingRect();
            validateROIFor(rect, mask);
            auto dstRoi = cv::Rect(xDstOffset, 0, rect.size().width, rect.size().height);
            mask(rect).copyTo(dst(dstRoi));
            xDstOffset += rect.size().width;
        }
    }

    static void resizeAndFitACenter(cv::Mat &src, const cv::Size &size, const cv::Scalar &background) {
        if (src.cols > src.rows) {
            cv::Mat resize = cv::Mat(src.cols, src.cols, src.type(), background);
            src.copyTo(resize(cv::Rect(0, (src.cols - src.rows) / 2, src.cols, src.rows)));
            src = resize;
        } else {
            cv::Mat resize = cv::Mat(src.rows, src.rows, src.type(), background);
            src.copyTo(resize(cv::Rect((src.rows - src.cols) / 2, 0, src.cols, src.rows)));
            src = resize;
        }
        cv::resize(src, src, size, cv::INTER_AREA);
    }

    static void extractAndStraighten(cv::Mat &src, cv::Mat &dest, const cv::RotatedRect &patchROI) {

        // obtain the bounding box of the desired patch
        cv::Rect boundingRect = patchROI.boundingRect();
        validateROIFor(boundingRect, src);
        auto width = patchROI.size.width;
        auto height = patchROI.size.height;
        auto angle = patchROI.angle;

        // crop out the bounding rectangle from the source image
        auto preCropImg = src(boundingRect);

        // the rotational center relative tot he pre-cropped image
        int cropMidX, cropMidY;
        cropMidX = boundingRect.width / 2;
        cropMidY = boundingRect.height / 2;

        // obtain the affine transform that maps the patch ROI in the image to the
        // dest patch image. The dest image will be an upright version.
        auto map_mat = cv::getRotationMatrix2D(cv::Point2f(cropMidX, cropMidY), angle, 1.0f);
        map_mat.at<double>(0, 2) += static_cast<double>(width / 2 - cropMidX);
        map_mat.at<double>(1, 2) += static_cast<double>(height / 2 - cropMidY);

        // rotate the pre-cropped image. The destination image will be
        // allocated by warpAffine()
        cv::warpAffine(preCropImg, dest, map_mat, cv::Size2i(width, height));
    }

    static bool mergeBlobs(cv::Mat &mask, std::vector<std::vector<cv::Point>> &contours) {
        int size = 0;
        do {
            size += 10;
            auto element = getStructuringElement(0, cv::Size(2 * size + 1, 2 * size + 1), cv::Point(size, size));
            dilate(mask, mask, element);
            findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        } while (contours.size() > 1 && size < 255);

        return contours.size() == 1;
    }

    static bool findFrameFromContours(int index, const std::vector<std::vector<cv::Point>> &contours,
                                      std::vector<std::vector<cv::Point>> &hulls, std::vector<cv::Point> &simplifiedHull) {
        convexHull(contours[index], hulls[index]);
        auto peri = cv::arcLength(hulls[index], true);

        double epsilon = 0.11;
        do {
            epsilon += 0.01;
            cv::approxPolyDP(hulls[index], simplifiedHull, epsilon * 0.1 * peri, true);
        } while (simplifiedHull.size() > 4 && epsilon < 10.0);
        hulls[index] = simplifiedHull;
        return simplifiedHull.size() == 4;
    }

    static cv::Rect orderAndSetCorners(const cv::Mat &src, const std::vector<cv::Point> &points,
                                       cv::Point2f *orderedPoints,
                                       cv::Point2f *straightPoints) {
        cv::Point topLeft;
        cv::Point topRight;
        cv::Point bottomRight;
        cv::Point bottomLeft;

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

        int width = std::max(topWidth, bottomWidth);
        int height = std::max(leftHeight, rightHeight);

        orderedPoints[0] = topLeft;
        orderedPoints[1] = topRight;
        orderedPoints[2] = bottomRight;
        orderedPoints[3] = bottomLeft;

        straightPoints[0] = cv::Point(0, 0);
        straightPoints[1] = cv::Point(width, 0);
        straightPoints[2] = cv::Point(width, height);
        straightPoints[3] = cv::Point(0, height);

        return cv::Rect(0, 0, width, height);
    }

    static void addSaltAndPepper(cv::Mat &dst, float pa = 0.05, float pb = 0.05) {
        cv::RNG rng;
        int amount1 = dst.rows * dst.cols * pa;
        int amount2 = dst.rows * dst.cols * pb;
        for (int counter = 0; counter < amount1; ++counter) {
            dst.at<uchar>(rng.uniform(0, dst.rows), rng.uniform(0, dst.cols)) = 0;

        }
        for (int counter = 0; counter < amount2; ++counter) {
            dst.at<uchar>(rng.uniform(0, dst.rows), rng.uniform(0, dst.cols)) = 255;
        }
    }

    int log(const char *fmt, ...) {
        int ret;
        va_list myargs;
        va_start(myargs, fmt);
        static char buffer[4096];
        ret = vsprintf(buffer, fmt, myargs);
        va_end(myargs);
        if (foo != nullptr) {
            foo(buffer);
        } else {
            std::cout << buffer << std::endl;
        }
        return ret;
    }
};