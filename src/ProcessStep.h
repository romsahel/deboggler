//
// Created by Roman SAHEL on 29/01/2022.
//

#ifndef DEBOGGLER_PROCESSSTEP_H
#define DEBOGGLER_PROCESSSTEP_H

#include "commons.h"

template<typename T>
bool trackbar(const char *label, const cv::Rect &window, T &value, T min, T max) {
    constexpr const char *format = (std::is_integral_v<T>) ? "%.0Lf" : "%.2Lf";
    auto &aBlock = cvui::internal::topBlock();
    bool hasChanged = cvui::trackbar(window.width - 25 - 5, &value, min, max, 1, format);
    cvui::text(aBlock.where, window.width - 25 + 5, aBlock.anchor.y - 45 + 10, label);
    return hasChanged;
}

struct ProcessStep {
    const char *filename;

    virtual ~ProcessStep() {

    }

    virtual void Process(const cv::Mat &src, cv::Mat &current) {
    }

    virtual const char *GUILabel() = 0;

    void DrawLabel(bool isLast) {
        cvui::text(GUILabel(), cvui::DEFAULT_FONT_SCALE, isLast ? 0xff0000 : 0xCECECE);
    }

    virtual bool DrawGUI(const cv::Rect &window) {
        return false;
    }
};

#endif //DEBOGGLER_PROCESSSTEP_H
