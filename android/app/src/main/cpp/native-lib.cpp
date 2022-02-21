#include <jni.h>
#include <android/log.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <functional>

#define TAG "Deboggler_Native"

#define FEEDFORWARD
#include "ProcessImage.h"


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



int JNICALL
Java_com_rsahel_deboggler_CameraFragment_deboggle(JNIEnv *env, jobject instance,
                                                  jlong srcAddr, jcharArray ptr
) {
    auto& deboggler = get_deboggler();
    Mat &current = *(Mat *) srcAddr;
    static bool initialize = true;
    if (initialize) {
        initialize = false;
        deboggler.logCallback = [](const char* fmt) {
            __android_log_print(ANDROID_LOG_INFO, TAG, "%s\n", fmt);
        };
        mask = cv::Mat::zeros(current.rows, current.cols, CV_8UC3);
    }

    jchar *body = env->GetCharArrayElements(ptr, nullptr);
    deboggler.guessedBoard = body;

    ProcessResult result = ProcessResult::PROCESS_FAILURE;
    try {
        cv::cvtColor(current, current, cv::COLOR_RGB2BGR);
        result = deboggler.Process(current, mask);
//    __android_log_print(ANDROID_LOG_INFO, TAG, "result: %d\n", result);
        if (result == ProcessResult::PROCESS_SUCCESS) {
            env->SetCharArrayRegion(ptr, 0, 16, deboggler.guessedBoard);
        }
        cv::cvtColor(current, current, cv::COLOR_BGR2RGB);
    }
    catch (cv::Exception &e) {
        __android_log_print(ANDROID_LOG_INFO, TAG, "exception caught: %s\n", e.what());
    }

//    if (result >= ProcessResult::Warped) {
//        if (deboggler.warpedMat.channels() == 1) {
//            cv::cvtColor(deboggler.warpedMat, deboggler.warpedMat, cv::COLOR_GRAY2RGB);
//        }
//        deboggler.warpedMat.copyTo(current(cv::Rect(0, 0, deboggler.warpedMat.cols, deboggler.warpedMat.rows)));
//    }

    return int(result);
}
}