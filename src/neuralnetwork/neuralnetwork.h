//
// Created by Roman SAHEL on 05/02/2022.
//

#ifndef DEBOGGLER_NEURALNETWORK_H
#define DEBOGGLER_NEURALNETWORK_H

#include <iostream>
#include <opencv2/core/core.hpp>

template<typename UnaryFunc, typename Mat>
Mat map(Mat &&input, UnaryFunc func) {
    auto outputPtr = (float *) (input.template begin<float>().ptr);
    auto outputPtrEnd = (float *) (input.template end<float>().ptr);
    for (; outputPtr != outputPtrEnd; ++outputPtr)
        *outputPtr = func(*outputPtr);
    return input;
}

#define print_mat(MAT) print(MAT, #MAT)

template<typename Mat>
auto &print(Mat &&input, const char *name) {
    return std::cout << name << " = " << std::endl << " " << input << std::endl << std::endl;
}

float sigmoid(float x) {
    return 1.0f / (1 + std::exp(-x));
}

float dsigmoid(float sigmoid) {
    // sigmoid(x) - (1.0f - sigmoid(x));
    return sigmoid - (1.0f - sigmoid);
}

struct NeuralNetwork {
    cv::Mat m_weightsInput;
    cv::Mat m_weightsHidden;

    cv::Mat m_biasInput;
    cv::Mat m_biasHidden;

    float learningRate = 0.01f;

    NeuralNetwork(int nbInputs, int nbHiddens, int nbOutputs)
            : m_weightsInput(cv::Mat::zeros(nbHiddens, nbInputs, CV_32FC1)),
              m_weightsHidden(cv::Mat::zeros(nbOutputs, nbHiddens, CV_32FC1)),
              m_biasInput(cv::Mat::zeros(nbHiddens, 1, CV_32FC1)),
              m_biasHidden(cv::Mat::zeros(nbOutputs, 1, CV_32FC1)) {
        cv::randu(m_weightsInput, -1.0f, 1.0f);
        cv::randu(m_weightsHidden, -1.0f, 1.0f);
        cv::randu(m_biasInput, -1.0f, 1.0f);
        cv::randu(m_biasHidden, -1.0f, 1.0f);
    }

    [[nodiscard]] cv::Mat feed_forward_hiddens(const cv::Mat& inputs) const {
        return map(cv::Mat((m_weightsInput * inputs) + m_biasInput), sigmoid);
    }

    [[nodiscard]] cv::Mat feed_forward_outputs(const cv::Mat& hiddens) const {
        return map(cv::Mat((m_weightsHidden * hiddens) + m_biasHidden), sigmoid);
    }

    [[nodiscard]] cv::Mat feed_forward(const cv::Mat& inputs) const {
        return feed_forward_outputs(feed_forward_hiddens(inputs));
    }

    void train(const cv::Mat& inputs, const cv::Mat& targets) {
        auto hiddens = feed_forward_hiddens(inputs);
        auto outputs = feed_forward_outputs(hiddens);

        // Outputs to Hiddens backpropagation
        cv::Mat outputErrors = targets - outputs;
        // calculate deltas outputs->hiddens: lr * Errors * (Outputs*(1-Outputs)) * transpose(Hiddens)
        auto deltaHiddens = learningRate * outputErrors * map(outputs.clone(), dsigmoid) * hiddens.t();

        m_weightsHidden += deltaHiddens;
        
        // Hiddens to Inputs backpropagation
        cv::Mat hiddenErrors = m_weightsHidden.t() * outputErrors;
        // calculate deltas hiddens->inputs: lr * Errors * (Hiddens*(1-Hiddens)) * transpose(Inputs)
        auto deltaInputs = learningRate * hiddenErrors * map(hiddens.clone(), dsigmoid) * inputs.t();

        m_weightsInput += deltaInputs;
        
    }
};
#endif //DEBOGGLER_NEURALNETWORK_H
