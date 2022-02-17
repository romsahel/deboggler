#include <jni.h>
#include <android/log.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <functional>

#define TAG "Deboggler_Native"

using namespace std;
using namespace cv;
#define FEEDFORWARD
#include "ProcessImage.h"


float alpha = 1.0f;
Mat mask;


Deboggler& get_deboggler() {
    static Deboggler deboggler;
    return deboggler;
}

extern "C" {

void JNICALL
Java_com_rsahel_deboggler_CameraFragment_configureNeuralNetwork(JNIEnv *env, jobject instance, jstring jstr) {
    __android_log_print(ANDROID_LOG_INFO, TAG, "configureNeuralNetwork\n");
    const char *path = env->GetStringUTFChars(jstr, nullptr);
    __android_log_print(ANDROID_LOG_INFO, TAG, "path to configuration: %s\n", path);
    get_deboggler().neuralNetwork.deserialize(path);
    __android_log_print(ANDROID_LOG_INFO, TAG, "configuration read\n");
    env->ReleaseStringUTFChars(jstr, path);
}

jstring JNICALL
Java_com_rsahel_deboggler_CameraFragment_deboggle(JNIEnv *env, jobject instance, jlong srcAddr) {

    Mat &current = *(Mat *) srcAddr;
    auto& deboggler = get_deboggler();

    static bool initialize = true;
    if (initialize) {
        initialize = false;
        deboggler.initializeArea(current);
        deboggler.foo = [](const char* fmt) {
            __android_log_print(ANDROID_LOG_INFO, TAG, "%s\n", fmt);
        };
        mask = cv::Mat::zeros(current.rows, current.cols, CV_8UC3);
    }
    ProcessResult result = ProcessResult::PROCESS_FAILURE;
    try {
        result = deboggler.Process(current, mask);
    }
    catch (cv::Exception &e) {
        __android_log_print(ANDROID_LOG_INFO, TAG, "exception caught: %s\n", e.what());
    }
//    if (result == ProcessResult::PROCESS_SUCCESS) {
//        float factor = (current.cols / 3) / deboggler.transformed.cols;
//        cv::resize(deboggler.transformed, deboggler.transformed, cv::Size(deboggler.transformed.cols * factor, deboggler.transformed.rows * factor));
//        deboggler.transformed.copyTo(current(cv::Rect(0, 0, deboggler.transformed.cols, deboggler.transformed.rows)));
//    }
//    else if (result >= ProcessResult::WARPED) {
//        cv::cvtColor(mask, mask, cv::COLOR_GRAY2BGR);
//        cv::bitwise_or(deboggler.transformed, mask, mask);
//        mask.copyTo(current(cv::Rect(0, 0, deboggler.transformed.cols, deboggler.transformed.rows)));
//    } else {
//        if (current.size() == mask.size()) {
////            if (mask.channels() == 1 && current.channels() == 3)
////                cv::cvtColor(mask, mask, cv::COLOR_GRAY2BGR);
////            cv::bitwise_or(current, mask, current);
////            current.copyTo(current, mask);
//        }
//        current = mask;
//    }

    return result == ProcessResult::PROCESS_SUCCESS ? env->NewStringUTF(deboggler.result.c_str()) : nullptr;
}
}