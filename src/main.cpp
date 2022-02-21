#include <iostream>
#include <opencv2/highgui/highgui.hpp>

#include "commons.h"
#include "FindWhiteBlobs.h"
#include "ExtractDies.h"
#include "Assembly.h"
#include "MergeWhiteBlobs.h"

#define FEEDFORWARD
#define WRITE_IMAGE

#include "../android/app/src/main/cpp/ProcessImage.h"

struct DebogglerStep : ProcessStep {
    Assembly &assembly;
    Deboggler deboggler;
    NeuralNetwork neuralNetwork;
    int maxStep = int(ProcessResult::PROCESS_SUCCESS);

    DebogglerStep(Assembly &assembly) : assembly(assembly) {
        neuralNetwork.deserialize("../neuralNetwork.bin");
    }

    const char *GUILabel() override { return "Deboggler Step"; }

    void Reset(const cv::Mat &src) {
        deboggler = Deboggler();
        deboggler.imageName = assembly.targets[assembly.sourceIndex];
        static uint16_t guessedBoard[16];
        deboggler.guessedBoard = guessedBoard;
        deboggler.neuralNetwork = neuralNetwork;
    }

    void Process(const cv::Mat &src, cv::Mat &current) override {
        if (assembly.sourceIndex != previousIndex && previousIndex < 0) {
            previousIndex = assembly.sourceIndex;
            Reset(src);
        }
        deboggler.maxStep = ProcessResult(maxStep);

        current = src;
        cv::Mat mask = cv::Mat::zeros(src.rows, src.cols, CV_8UC3);

        auto result = deboggler.Process(current, mask);
//        if (result >= ProcessResult::WARPED) {
//            if (deboggler.warpedMat.channels() == 1) {
//                cv::cvtColor(deboggler.warpedMat, deboggler.warpedMat, cv::COLOR_GRAY2RGB);
//            }
//            deboggler.warpedMat.copyTo(current(cv::Rect(0, 0, deboggler.warpedMat.cols, deboggler.warpedMat.rows)));
//        }
        if (result == ProcessResult::PROCESS_SUCCESS) {
            bool incorrect = false;
            std::string guess;
            for (int i = 0; i < 16 && !incorrect; ++i) {
                incorrect |= deboggler.guessedBoard[i] != deboggler.imageName[i];
                guess.push_back((char) deboggler.guessedBoard[i]);
            }
            if (incorrect) {
                std::cout << "Incorrect: " << deboggler.imageName << " (found: " << guess << ")" << std::endl;
            }
        } else {
            std::cout << "Failure" << std::endl;
        }

        int windowSize = 256;
        if (mask.size().width > windowSize) {
            cv::resize(mask, mask, cv::Size(windowSize, windowSize / mask.size().aspectRatio()));
        }
        cv::imshow("mask", mask);
        cv::moveWindow("mask", assembly.inspectorWidth + 256, 200);
        if (result >= ProcessResult::Warped) {
            if (deboggler.warpedMat.size().width > 9128) {
                cv::resize(deboggler.warpedMat, mask, cv::Size(256, 256 / mask.size().aspectRatio()));
                cv::imshow("warpedMat", mask);
            } else {
                cv::imshow("warpedMat", deboggler.warpedMat);
            }
            cv::moveWindow("warpedMat", assembly.inspectorWidth + 256 + 256, 200);
        } else {
            cv::destroyWindow("warpedMat");
        }
    }

    bool DrawGUI(const cv::Rect &window) override {
        bool hasChanged = false;
        hasChanged |= trackbar("Step", window, maxStep, int(ProcessResult::PROCESS_FAILURE) + 1, int(ProcessResult::PROCESS_SUCCESS));
        hasChanged |= trackbar("sat", window, deboggler.low_s, 0, 255);
        hasChanged |= trackbar("hue", window, deboggler.high_h, 0, 255);
        hasChanged |= trackbar("canny", window, deboggler.canny_threshold, 0, 255);

        return hasChanged;
    }

    int previousIndex = -1;
};


int main(int argc, const char *argv[]) {
    Assembly assembly;
    if (false) {
        assembly.push_back(new FindWhiteBlobs());
        assembly.push_back(new MergeWhiteBlobs());
        assembly.push_back(new ExtractDies(assembly));
    } else {
        assembly.showMask = false;
        assembly.push_back(new DebogglerStep(assembly));
    }

    assembly.init();

//    for (assembly.sourceIndex = 0; assembly.sourceIndex < assembly.sources.size(); ++assembly.sourceIndex) {
//        assembly.load(cv::imread(assembly.sources[assembly.sourceIndex]));
//        assembly.update();
//    }
//    assembly.sourceIndex = 0;
//    assembly.load(cv::imread(assembly.sources[assembly.sourceIndex]));

    while (true) {
        assembly.update();
        if (assembly.draw()) {
            break;
        }
        cvui::update();
        assembly.show();
    }

    return 0;
}

