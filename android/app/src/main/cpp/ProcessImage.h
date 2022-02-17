#pragma once

#include "/Library/dev/rsahel/deboggler-repo/src/neuralnetwork/neuralnetwork.h"

enum class ProcessResult {
    BLOBS_NOT_FOUND,
    BLOBS_NOT_MERGED,
    FRAME_NOT_FOUND,
    PROCESS_FAILURE, // End of failures

    BLOBS_FOUND,
    BLOBS_MERGED,
    FRAME_FOUND,
    CORNERS_FOUND,
    WARPED,
    DIES_CONTOURED,
    DIES_CLOSED,
    PROCESS_SUCCESS // End of successes
};

int max_process_steps() {
    return int(ProcessResult::PROCESS_SUCCESS) - int(ProcessResult::PROCESS_FAILURE) - 1;
}

bool is_failure(ProcessResult result) {
    return result <= ProcessResult::PROCESS_FAILURE;
}

bool is_success(ProcessResult result) {
    return result > ProcessResult::PROCESS_FAILURE;
}

template<typename T>
static bool approximately(T a, T b, T epsilon) {
    return std::abs(b - a) < epsilon;
}

struct Deboggler {
    static constexpr int baseSensitivity = 150;
    int sensitivity = baseSensitivity;
    int sensitivity2 = baseSensitivity;
    int denoisingStrength = 1;
    int thresholdv = 150;
    std::vector<std::vector<cv::Point>> contours;
    std::vector<std::vector<cv::Point>> hull{1};
    std::vector<cv::Point> simplifiedHull;
    std::vector<cv::Rect> boundRect;
    std::vector<cv::Rect> boundRect2;
    int maxStep = max_process_steps();
    std::string directoryName;

    cv::Rect safeArea;
    int idealArea;
    int maxArea, minArea;

    cv::Mat transformed;
    cv::Point2f orderedPoints[4];
    cv::Point2f straightPoints[4];
    std::string result;

    NeuralNetwork neuralNetwork;
    void (*foo)(const char*);

    void initializeArea(const cv::Mat &src) {
        int w = src.cols, h = src.rows;
        const float safeFactor = 0.05f;
        safeArea = cv::Rect(0, 0, w, h);
        idealArea = w * h / 4;
        maxArea = idealArea, minArea = idealArea * 0.025f;
    }

    ProcessResult Process(cv::Mat &src, cv::Mat &mask) {
        int currentStep = 0;
        contours.clear();
        currentStep++;
        auto blobsFound = findBlobs(src, mask, sensitivity, true);

        // Draw blobs filled
        mask = 0;
        for (size_t i = 0; i < contours.size(); i++) {
            drawContours(mask, contours, (int) i, i < 16 ? 255 : 127, -1);
        }
        
        if (!blobsFound) {
            return ProcessResult::BLOBS_NOT_FOUND;
        }
        if (currentStep > maxStep)
            return ProcessResult::BLOBS_FOUND;

        currentStep++;
        if (!mergeBlobs(mask)) {
            return ProcessResult::BLOBS_NOT_MERGED;
        }
        if (currentStep > maxStep)
            return ProcessResult::BLOBS_MERGED;

        currentStep++;
        if (!findFrame(0)) {
            return ProcessResult::FRAME_NOT_FOUND;
        }
        hull[0] = simplifiedHull;
        if (currentStep > maxStep) 
        {
            mask = 0;
            drawContours(mask, hull, 0, 255, 15);
            for (int i = 0; i < 4; ++i) {
                cv::circle(mask, simplifiedHull[i], 16 + 4.0f * i, cv::Scalar(255, 255, 255), -1);
            }
            cv::cvtColor(mask, mask, cv::COLOR_GRAY2BGR);
            cv::bitwise_or(src, mask, mask);
            return ProcessResult::FRAME_FOUND;
        }

        currentStep++;
        mask = 0;
        orderAndSetCorners(src, mask, simplifiedHull);
        if (currentStep > maxStep) {
            mask = 0;
            for (int i = 0; i < 4; ++i) {
                cv::circle(mask, orderedPoints[i], 16 + 4.0f * i, cv::Scalar(255, 255, 255), -1);
            }
            return ProcessResult::CORNERS_FOUND;
        }

        currentStep++;
        auto transform = cv::getPerspectiveTransform(orderedPoints, straightPoints);
        cv::warpPerspective(src, transformed, transform, transformed.size());
        mask = cv::Mat::zeros(transformed.rows, transformed.cols, mask.type());
        if (currentStep > maxStep)
            return ProcessResult::WARPED;

        currentStep++;
//        cv::cvtColor(transformed, mask, cv::COLOR_BGR2GRAY);
//        cv::threshold(mask, mask, thresholdv, 255, cv::THRESH_BINARY);
        findBlobs(transformed, mask, sensitivity, true);
        unsigned long contoursCount = contours.size();
        if (currentStep > maxStep || contoursCount < 16) {
            return ProcessResult::DIES_CLOSED;
        }
        
        if (currentStep > maxStep) 
        {
            mask = 0;
            for (size_t i = 0; i < contours.size(); i++) {
                drawContours(mask, contours, (int) i, 255, -1);
            }
            return ProcessResult::DIES_CLOSED;
        }

        if (hull.size() < contoursCount) {
            hull.resize(contoursCount);
        }
        if (boundRect.size() < contoursCount) {
            boundRect.resize(contoursCount);
        }
        for (size_t i = 0; i < contoursCount; i++) {
            approxPolyDP(contours[i], hull[i], 3, true);
            boundRect[i] = boundingRect(hull[i]);
        }
        
        auto white = cv::Scalar(255, 255, 255);
        int maxWidth = 28, maxHeight = 28;

        std::sort(std::begin(boundRect), std::begin(boundRect) + contoursCount, [](auto &a, auto &b) {
            return a.y < b.y;
        });
        for (int i = 0; i < 4; ++i) {
            std::sort(std::begin(boundRect) + i * 4, std::begin(boundRect) + (i + 1) * 4, [](auto &a, auto &b) {
                return a.x < b.x;
            });
        }

        transformed = cv::Mat(cv::Size(maxWidth * 4, maxHeight * 4), CV_8UC1, 255);
        int copyX = 0, copyY = 0;
        result.clear();
#ifdef WRITE_IMAGE
        auto folder = "../output/" + directoryName + '/';
        if (!directoryName.empty()) {
            if (!std::filesystem::is_directory(folder) || !std::filesystem::exists(folder)) { // Check if src folder exists
                std::filesystem::create_directory(folder); // create src folder
            }
        }
#endif
        for (size_t i = 0; i < 16; i++) {
            cv::Mat mat = mask(boundRect[i]);
            cv::RotatedRect box;
            auto valid = findDiceBoundingBox(mat, box);
//            mat = 0;
//            for (int j = 0; j < contours.size(); ++j) {
//                drawContours(mat, contours, j, 255, 15);
//            }
//            continue;
//            if (valid)
                valid = extractPatchFromOpenCVImage(mat, mat, box);
            if (!valid)
            {
#ifdef WRITE_IMAGE
                log("error: %c", directoryName[i]);
#else
                log("invalid");
#endif
//                break;
            }
//            cv::Mat kernelOne = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(denoisingStrength * 2 + 1, denoisingStrength * 2 + 1));
//            cv::morphologyEx(mat, mat, cv::MORPH_CLOSE, kernelOne);

            if (mat.cols > mat.rows) {
                cv::Mat resize = cv::Mat(mat.cols, mat.cols, mat.type(), 255);
                mat.copyTo(resize(cv::Rect(0, (mat.cols - mat.rows) / 2, mat.cols, mat.rows)));
                mat = resize;
            } else {
                cv::Mat resize = cv::Mat(mat.rows, mat.rows, mat.type(), 255);
                mat.copyTo(resize(cv::Rect((mat.rows - mat.cols) / 2, 0, mat.cols, mat.rows)));
                mat = resize;
            }
            cv::resize(mat, mat, cv::Size(transformed.cols / 4, transformed.rows / 4), cv::INTER_AREA);

            mat.copyTo(transformed(cv::Rect(copyX, copyY, mat.cols, mat.rows)));
            if (i == 3 || i == 7 || i == 11) {
                copyX = 0;
                copyY += mat.rows;
            } else {
                copyX += mat.cols;
            }
#ifdef WRITE_IMAGE
            if (!directoryName.empty()) {
                int img_index = 0;
                for (int j = 0; j < 4; ++j, img_index++) {
                    auto filename = folder + std::to_string(i) + '_' + std::to_string(img_index) + '_' + directoryName[i] + ".jpg";
                    imwrite(filename, mat);
                    cv::rotate(mat, mat, cv::ROTATE_90_CLOCKWISE);
                }
//                addSaltAndPepper(mat);
//                for (int j = 0; j < 4; ++j, img_index++) {
//                    auto filename = folder + std::to_string(i) + '_' + std::to_string(img_index) + '_' + directoryName[i] + ".jpg";
//                    imwrite(filename, mat);
//                    cv::rotate(mat, mat, cv::ROTATE_90_CLOCKWISE);
//                }
            }
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
//            log("guessed: %c", guessedChar);
            result += guessedChar;
#endif
        }

        return ProcessResult::PROCESS_SUCCESS;
    }
    bool extractPatchFromOpenCVImage( cv::Mat& src, cv::Mat& dest, cv::RotatedRect patchROI) {

        // obtain the bounding box of the desired patch
        cv::Rect boundingRect = patchROI.boundingRect();
        auto width = patchROI.size.width;
        auto height = patchROI.size.height;
        auto angle = patchROI.angle;

        // check if the bounding box fits inside the image
        if ( boundingRect.x >= 0 && boundingRect.y >= 0 &&
             (boundingRect.x+boundingRect.width) < src.cols &&
             (boundingRect.y+boundingRect.height) < src.rows ) {

            // crop out the bounding rectangle from the source image
            auto preCropImg = src(boundingRect);

            // the rotational center relative tot he pre-cropped image
            int cropMidX, cropMidY;
            cropMidX = boundingRect.width/2;
            cropMidY = boundingRect.height/2;

            // obtain the affine transform that maps the patch ROI in the image to the
            // dest patch image. The dest image will be an upright version.
            auto map_mat = cv::getRotationMatrix2D(cv::Point2f(cropMidX, cropMidY), angle, 1.0f);
            map_mat.at<double>(0,2) += static_cast<double>(width/2 - cropMidX);
            map_mat.at<double>(1,2) += static_cast<double>(height/2 - cropMidY);

            // rotate the pre-cropped image. The destination image will be
            // allocated by warpAffine()
            cv::warpAffine(preCropImg, dest, map_mat, cv::Size2i(width,height));

            return true;
        } // if
        else {
            return false;
        } // else
    } // extractPatch

    bool findBlobs(cv::Mat &src, cv::Mat &dst, int &parameter, bool useSafeArea) {
        cv::cvtColor(src, src, cv::COLOR_BGR2HSV);
        extractAndFilterBlobContours(src, dst, parameter, useSafeArea);
        if (contours.size() != 16) {
            for (parameter = baseSensitivity; parameter >= 64; parameter--) {
                extractAndFilterBlobContours(src, dst, parameter, useSafeArea);
                if (contours.size() == 16) {
                    break;
                }
            }
        }
        cv::cvtColor(src, src, cv::COLOR_HSV2BGR);
        return contours.size() == 16;
    }
    
    struct Blabla {
        int score = 0;
        cv::RotatedRect rect;
        std::vector<cv::Point> contour;
    };

    void extractAndFilterBlobContours(cv::Mat &src, cv::Mat &dst, int s, bool useSafeArea) {

        cv::Mat mask;
        cv::cvtColor(src, mask, cv::COLOR_HSV2BGR);
        cv::cvtColor(mask, mask, cv::COLOR_BGR2GRAY);
        blur(mask, mask, Size(3, 3));
        Canny(mask, mask, 255, 0, 3);
        cv::Mat kernelOne = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(denoisingStrength * 2 + 1, denoisingStrength * 2 + 1));
        cv::dilate(mask, mask, kernelOne);
        cv::bitwise_not(mask, mask);

        cv::Vec3b lower(0, 0, 255 - s);
        cv::Vec3b upper(255, s, 255);
        cv::inRange(src, lower, upper, dst);
        cv::threshold(dst, dst, 127, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        cv::bitwise_and(dst, mask, dst);
        
        findContours(dst, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        std::vector<Blabla> scores;
        cv::Point2f barycenter;
        for (int i = 0; i < contours.size(); ++i) {
            auto rect = minAreaRect(contours[i]);
            bool isInvalid = false;
            isInvalid = isInvalid || rect.size.width > safeArea.width / 3;
            isInvalid = isInvalid || rect.size.height > safeArea.height / 3;
            isInvalid = isInvalid || rect.size.width < safeArea.width / 16;
            isInvalid = isInvalid || rect.size.height < safeArea.height / 16;
            if (!isInvalid) {
                auto& score = scores.emplace_back();
                score.rect = rect;
                score.contour = contours[i];
                score.score += contours[i].size() * 10;
                score.score += std::abs(1.0f - float (rect.size.aspectRatio())) * 10000.0f;
                barycenter += rect.center;
            }
        }
        if (scores.size() >= 16) 
        {
            barycenter /= float(scores.size());
            for (int i = 0; i < scores.size(); ++i) {
                scores[i].score += cv::norm(scores[i].rect.center - barycenter);
            }
            std::sort(std::begin(scores), std::end(scores), [](auto &a, auto &b) {
                return a.score < b.score;
            });
            contours.clear();
            for (int i = 0; i < scores.size(); ++i) {
                if (scores[i].score > 20000)
                    break;
                contours.push_back(scores[i].contour);
            }
        }
    }

    bool isBlobValid(const std::vector<cv::Point> &o, bool useSafeArea) const {
        cv::RotatedRect rect = minAreaRect(o);
        bool remove = false;
        remove = remove || o.size() > 2000;
        remove = remove || rect.size.width > safeArea.width / 4;
        remove = remove || rect.size.height > safeArea.height / 4;
        remove = remove || rect.size.width < safeArea.width / 16;
        remove = remove || rect.size.height < safeArea.height / 16;
        remove = remove || (!approximately(rect.size.width / rect.size.height, 1.0f, 0.2f));
        return remove;
    }

    // Dilate blobs until it forms one big blob
    bool mergeBlobs(cv::Mat &mask) {
        int size = 0;
        do {
            size += 10;
            auto element = getStructuringElement(0, cv::Size(2 * size + 1, 2 * size + 1), cv::Point(size, size));
            dilate(mask, mask, element);
            findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        } while (contours.size() > 1 && size < 255);

        return contours.size() == 1;
    }

    bool findFrame(int index) {
        convexHull(contours[index], hull[index]);
        auto peri = cv::arcLength(hull[index], true);

        double epsilon = 0.11;
        do {
            epsilon += 0.01;
            cv::approxPolyDP(hull[index], simplifiedHull, epsilon * 0.1 * peri, true);
        } while (simplifiedHull.size() > 4 && epsilon < 10.0);
        return simplifiedHull.size() == 4;
    }

    void orderAndSetCorners(const cv::Mat &src, cv::Mat &current, const std::vector<cv::Point> &points) {
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

        transformed = cv::Mat::zeros(height, width, src.type());
    }

    bool findDiceBoundingBox(cv::Mat &mat, cv::RotatedRect& result) {
        findContours(mat, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
        std::vector<Blabla> scores;
        cv::Point2f barycenter;
        int h = mat.cols, w = mat.rows;
        for (int i = 0; i < contours.size(); ++i) {
            auto rect = minAreaRect(contours[i]);
            auto bb = rect.boundingRect2f();
            bool isInvalid = false;
            isInvalid = isInvalid || bb.size().width > w * 0.95f;
            isInvalid = isInvalid || bb.size().height > h * 0.95f;
            isInvalid = isInvalid || bb.size().width < w  * 0.05f;
            isInvalid = isInvalid || bb.size().height < h  * 0.05f;
            if (!isInvalid) {
                auto& score = scores.emplace_back();
                score.rect = rect;
                score.contour = contours[i];
                score.score += -score.rect.size.area();
                barycenter += rect.center;
            }
        }
//        log("findDiceBoundingBox: %lu // %lu", contours.size(), scores.size());
        if (scores.size() == 0){
            return false;   
        }


        std::sort(std::begin(scores), std::end(scores), [](auto &a, auto &b) {
            return a.score < b.score;
        });
        
        contours.clear();
        contours.push_back(scores[0].contour);
        auto scaledRect = scores[0].rect.boundingRect();
        int rwidth  = scaledRect.width;
        int rheight = scaledRect.height;
        scaledRect.width = std::round(scaledRect.width * 1.2f);
        scaledRect.height = std::round(scaledRect.height * 1.2f);
        scaledRect.x += (rwidth - scaledRect.width) / 2;
        scaledRect.y += (rheight - scaledRect.height) / 2;

        for (int i = 1; i < scores.size(); ++i) {
            static cv::Point2f corners[4];
            scores[i].rect.points(corners);
            bool valid= false;
            for (int j = 0; j < 4 && !valid; ++j) {
                valid = scaledRect.contains(corners[j]); 
            }
            if (valid) { 
                contours.push_back(scores[i].contour); 
            }
        }

        for (int i = contours.size() - 1; i > 0; --i) {
            contours[0].insert(contours[0].end(), contours[i].begin(), contours[i].end());
            contours.pop_back();
        }
        result = minAreaRect(contours[0]);
        return true;
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

    int log(const char *fmt, ...)
    {
        int ret;
        va_list myargs;
        va_start(myargs, fmt);
        static char buffer[4096];
        ret = vsprintf(buffer, fmt, myargs);
        va_end(myargs);
        if (foo != nullptr) foo(buffer);
        return ret;
    }
};