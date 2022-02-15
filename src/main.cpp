#include <iostream>
#include <opencv2/highgui/highgui.hpp>

#include "commons.h"
#include "FindWhiteBlobs.h"
#include "ExtractDies.h"
#include "Assembly.h"
#include "MergeWhiteBlobs.h"

#define WRITE_IMAGE
#include "/Library/dev/rsahel/deboggler-android/app/src/main/cpp/ProcessImage.h"

struct DebogglerStep : ProcessStep {
    Assembly &assembly;
    Deboggler deboggler;
    NeuralNetwork neuralNetwork;

    DebogglerStep(Assembly &assembly) : assembly(assembly) {
        neuralNetwork.deserialize("neuralNetwork.bin");
    }

    const char *GUILabel() override { return "Deboggler Step"; }

    void Reset(const cv::Mat &src) {
        deboggler = Deboggler();
        deboggler.directoryName = assembly.targets[assembly.sourceIndex];
        deboggler.initializeArea(src);
        deboggler.neuralNetwork = neuralNetwork;
    }

    void Process(const cv::Mat &src, cv::Mat &current) override {
        if (assembly.sourceIndex != previousIndex) {
            previousIndex = assembly.sourceIndex;
            Reset(src);
        }
        current = src;
        cv::Mat mask = cv::Mat::zeros(src.rows, src.cols, CV_8UC3);
        auto result = deboggler.Process(current, mask);
//        if (result == ProcessResult::PROCESS_SUCCESS) {
//            assembly.showMask = false;
//            float factor = (src.cols / 3) / deboggler.transformed.cols;
//            cv::resize(deboggler.transformed, deboggler.transformed, cv::Size(deboggler.transformed.cols * factor, deboggler.transformed.rows * factor));
//            deboggler.transformed.copyTo(current(cv::Rect(0, 0, deboggler.transformed.cols, deboggler.transformed.rows)));
//            std::cout << deboggler.result << std::endl;
//        }
//        else if (result >= ProcessResult::WARPED) {
//            assembly.showMask = false;
//            cv::cvtColor(mask, mask, cv::COLOR_GRAY2BGR);
//            cv::bitwise_or(deboggler.transformed, mask, mask);
//            mask.copyTo(current(cv::Rect(0, 0, deboggler.transformed.cols, deboggler.transformed.rows)));
//        } else {
////            assembly.showMask = true;
//            current = mask;
//        }

        if (deboggler.result != deboggler.directoryName) {
            std::cout << "Incorrect: " << deboggler.directoryName << " (found: " << deboggler.result << ")" << std::endl;
        }
        cv::resize(mask, mask, cv::Size (256, 256 / mask.size().aspectRatio()));
        cv::imshow("mask", mask);
        cv::moveWindow("mask", assembly.inspectorWidth + 256, 200);
        if (result >= ProcessResult::WARPED) {
            if (deboggler.transformed.size().width > 128) {
                cv::resize(deboggler.transformed, mask, cv::Size(256, 256 / mask.size().aspectRatio()));
                cv::imshow("transformed", mask);
            } else {
                cv::imshow("transformed", deboggler.transformed);
            }
            cv::moveWindow("transformed", assembly.inspectorWidth + 256 + 256, 200);
        }
    }

    bool DrawGUI(const cv::Rect &window) override {
        bool hasChanged = false;
        hasChanged |= trackbar("Step", window, deboggler.maxStep, 0, max_process_steps());
        hasChanged |= trackbar("sensitivity", window, deboggler.sensitivity, 0, 255);
        hasChanged |= trackbar("sensitivity2", window, deboggler.sensitivity2, 0, 255);
        hasChanged |= trackbar("denoising strength", window, deboggler.denoisingStrength, 0, 16);
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

//    assembly.sourceIndex = 11;
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

