//
// Created by Roman SAHEL on 30/01/2022.
//

#ifndef DEBOGGLER_ASSEMBLY_H
#define DEBOGGLER_ASSEMBLY_H

#include "ProcessStep.h"

constexpr auto windowName = "Deboggler";
constexpr auto uiWindowName = "UI";

struct MaskStep : ProcessStep {
    const char *GUILabel() override { return "Mask"; }

    void Process(const cv::Mat &src, cv::Mat &current) override {
        if (src.size() == current.size()) {
            if (current.channels() == 1 && src.channels() == 3)
                cv::cvtColor(current, current, cv::COLOR_GRAY2BGR);
            cv::bitwise_or(src, current, current);
        }
    }
};

struct Assembly {
    std::vector<ProcessStep *> steps;
    ProcessStep *maskStep = new MaskStep();
    std::string filename;
    std::vector<const char *> sources = {
            "/Library/dev/deboggler/images/boggle-0.jpg",
            "/Library/dev/deboggler/images/boggle-1.jpg",
            "/Library/dev/deboggler/images/boggle-2.jpg",
            "/Library/dev/deboggler/images/boggle-3.jpg",
    };

    cv::Mat src;
    cv::Mat dst;
    cv::Mat uiFrame;

    bool firstShow = true;
    int inspectorWidth = 200;
    int inspectorHeight = 600;

    bool showMask = true;
    int maxStep = 0;
    int sourceIndex = 0;
    bool hasChanges = true;

    void init() {
        cvui::init(uiWindowName, 20);
        load(cv::imread(sources[sourceIndex]));
    }

    ~Assembly() {
        for (int i = 0; i < steps.size(); ++i) {
            delete steps[i];
        }
        delete maskStep;
    }

    void load(const cv::Mat &source) {
        src = source.clone();
        dst = src.clone();

        uiFrame = cv::Mat(inspectorHeight, inspectorWidth, src.type());

        filename = std::to_string(sourceIndex);
        for (int i = 0; i < maxStep; ++i) {
            steps[i]->filename = filename.c_str();
        }

        hasChanges = true;
    }

    void push_back(ProcessStep *step) {
        steps.push_back(step);
        maxStep = steps.size();
    }

    void update() {
        if (hasChanges) {
            src.copyTo(dst);
            for (int i = 0; i < maxStep; ++i) {
                steps[i]->Process(src, dst);
            }
            if (showMask) {
                maskStep->Process(src, dst);
            }
            hasChanges = false;
        }
    }

    bool draw() {

        auto contentRect = cvui::window(uiFrame, 0, 0, inspectorWidth, inspectorHeight, "Process");
        cvui::pad(contentRect, 10);
        cvui::beginColumn(uiFrame, contentRect.x, contentRect.y, -1, -1, 0);

        if (trackbar("Images", contentRect, sourceIndex, 0, (int) (sources.size() - 1))) {
            load(cv::imread(sources[sourceIndex]));
        }

        if (trackbar("Steps", contentRect, maxStep, 0, (int) steps.size())) {
            hasChanges = true;
        }
        hasChanges |= cvui::checkbox("Show mask", &showMask);

        for (int i = 0; i < steps.size(); ++i) {
            steps[i]->DrawLabel(i == maxStep - 1);
            hasChanges |= steps[i]->DrawGUI(contentRect);
            cvui::space(10);
        }

        cvui::endColumn();
        return false;
    }

    void show() {
        cv::imshow(uiWindowName, uiFrame);
        cv::imshow(windowName, dst);

        if (firstShow) {
            cv::moveWindow(uiWindowName, 0, 200);
            cv::moveWindow(windowName, inspectorWidth, 200);
            firstShow = false;
        }
    }
};

#endif //DEBOGGLER_ASSEMBLY_H
