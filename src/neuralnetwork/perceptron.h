//
// Created by Roman SAHEL on 05/02/2022.
//

#ifndef DEBOGGLER_PERCEPTRON_H
#define DEBOGGLER_PERCEPTRON_H

#include <iostream>
#include <opencv2/core/core.hpp>

constexpr int width = 800;
constexpr int height = 600;

cv::Point remap(const cv::Point2f &point) {
    return {static_cast<int>((1.0f + point.x) * 0.5f * width), height - static_cast<int>((1.0f + point.y) * 0.5f * height)};
}

float line(float x) {
    return 0.2f * x + 0.1f;
}

struct Perceptron {
    std::vector<float> weights;
    float learningRate = 0.0001f;

    Perceptron(int nbInputs) : weights(nbInputs) {
        static auto random = cv::RNG();
        for (int i = 0; i < nbInputs; ++i) {
            weights.push_back(random.uniform(-1.0f, 1.0f));
        }
    }

    int activate(float sum) const {
        return sum >= 0.0f ? 1 : -1;
    }

    int guess(const std::vector<float> &inputs) const {
        float sum = 0.0f;
        for (int i = 0; i < inputs.size(); ++i) {
            sum += inputs[i] * weights[i];
        }
        return activate(sum);
    }

    void train(const std::vector<float> &inputs, int target) {
        int result = guess(inputs);
        auto error = target - result;
        for (int i = 0; i < weights.size(); ++i) {
            weights[i] += error * inputs[i] * learningRate;
        }
    }


    float guessed_line(float x) {
        return -(weights[0] / weights[1]) * x - (weights[2] / weights[1]);
    }

    void draw(cv::Mat &canvas) {
        // ax + b = w0/w1x + w1 + bw2 = 0 => y = -w0x-w2/w1
        auto pt0 = remap(cv::Point2f(-1, guessed_line(-1)));
        auto pt1 = remap(cv::Point2f(1, guessed_line(1)));
        cv::line(canvas, pt0, pt1, cv::Scalar(255, 0, 255));
    }
};


#endif //DEBOGGLER_PERCEPTRON_H
