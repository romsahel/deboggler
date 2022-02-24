//
// Created by Roman SAHEL on 05/02/2022.
//

#ifndef DEBOGGLER_NEURALNETWORK_H
#define DEBOGGLER_NEURALNETWORK_H

#include <iostream>
#include <fstream>

#include <opencv2/core/core.hpp>

#include "serialization.h"

template<typename UnaryFunc, typename Mat>
Mat matmap(Mat &&input, UnaryFunc func) {
    auto outputPtr = (float *) (input.template begin<float>().ptr);
    auto outputPtrEnd = (float *) (input.template end<float>().ptr);
    for (; outputPtr != outputPtrEnd; ++outputPtr)
        *outputPtr = func(*outputPtr);
    return input;
}

#define print_mat(MAT) print(MAT, #MAT)
template<typename Mat>
auto &print(Mat &&input, const char *name) {
    return std::cout << name << " = " << std::endl << " " << input << std::endl;
}

float sigmoid(float x) {
    return 1.0f / (1 + std::exp(-x));
}

float dsigmoid(float sigmoid) {
    return sigmoid * (1.0f - sigmoid);
}

struct TrainingData {
    cv::Mat inputs;
    cv::Mat targets;
};

struct NeuralNetwork {
    cv::Mat m_weightsInputToHidden;
    cv::Mat m_weightsHiddenToOutput;

    cv::Mat m_biasInputToHidden;
    cv::Mat m_biasHiddenToOutput;

    float learningRate = 0.05f;

    NeuralNetwork() {}

    NeuralNetwork(int nbInputs, int nbHiddens, int nbOutputs)
            : m_weightsInputToHidden(cv::Mat::zeros(nbHiddens, nbInputs, CV_32FC1)),
              m_weightsHiddenToOutput(cv::Mat::zeros(nbOutputs, nbHiddens, CV_32FC1)),
              m_biasInputToHidden(cv::Mat::zeros(nbHiddens, 1, CV_32FC1)),
              m_biasHiddenToOutput(cv::Mat::zeros(nbOutputs, 1, CV_32FC1)) {
        cv::randu(m_weightsInputToHidden, -1.0f, 1.0f);
        cv::randu(m_weightsHiddenToOutput, -1.0f, 1.0f);
        cv::randu(m_biasInputToHidden, -1.0f, 1.0f);
        cv::randu(m_biasHiddenToOutput, -1.0f, 1.0f);
    }

    [[nodiscard]] cv::Mat feed_forward_to_hiddens(const cv::Mat &inputs) const {
        return matmap(cv::Mat((m_weightsInputToHidden * inputs) + m_biasInputToHidden), sigmoid);
    }

    [[nodiscard]] cv::Mat feed_forward_to_outputs(const cv::Mat &hiddens) const {
        return matmap(cv::Mat((m_weightsHiddenToOutput * hiddens) + m_biasHiddenToOutput), sigmoid);
    }

    [[nodiscard]] cv::Mat feed_forward(const cv::Mat &inputs) const {
        return feed_forward_to_outputs(feed_forward_to_hiddens(inputs));
    }

//    void train(std::vector<TrainingData> &trainingData, int epochs, int batchSize, float learningRate) {
//        auto n = trainingData.size();
//        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
//        for (int i = 0; i < epochs; ++i) {
//            std::shuffle(trainingData.begin(), trainingData.end(), std::default_random_engine(seed));
//            auto batchBegin = trainingData.begin();
//            while (batchBegin != trainingData.end()) {
//                auto batchEnd = std::distance(batchBegin, trainingData.end()) < batchSize ? batchBegin + batchSize : trainingData.end();
//                train(batchBegin, batchEnd);
//                batchBegin = batchEnd;
//            }
//        }
//    }
//
//    void train(std::vector<TrainingData>::const_iterator begin, std::vector<TrainingData>::const_iterator end) {
//        while (begin != end) {
//            begin++;
//        }
//    }
//
//    std::tuple<cv::Mat, cv::Mat> backpropagate(const TrainingData& trainingData) {
//        return {};
//    }

    float train(const cv::Mat &inputs, const cv::Mat &targets) {
        auto hiddens = feed_forward_to_hiddens(inputs);
        auto outputs = feed_forward_to_outputs(hiddens);

        // Outputs to Hiddens backpropagation
        auto errorOutputToHidden = targets - outputs;
        // calculate deltas outputs->hiddens: lr * Errors * (Outputs*(1-Outputs)) * transpose(Hiddens)
        auto gradientOutputToHidden = learningRate * errorOutputToHidden.mul(matmap(outputs.clone(), dsigmoid));
        auto deltaOutputToHidden = gradientOutputToHidden * hiddens.t();
        m_weightsHiddenToOutput += deltaOutputToHidden;
        m_biasHiddenToOutput += gradientOutputToHidden;

        // Hiddens to Inputs backpropagation
        cv::Mat errorHiddenToInput = m_weightsHiddenToOutput.t() * errorOutputToHidden;
        // calculate deltas hiddens->inputs: lr * Errors * (Hiddens*(1-Hiddens)) * transpose(Inputs)
        auto gradientHiddenToInput = learningRate * errorHiddenToInput.mul(matmap(hiddens.clone(), dsigmoid));
        auto deltaHiddenToInput = gradientHiddenToInput * inputs.t();
        m_weightsInputToHidden += deltaHiddenToInput;
        m_biasInputToHidden += gradientHiddenToInput;

        return norm(errorOutputToHidden);
    }

    void serialize(const char *path) const {
        std::ofstream fs(path, std::ios::out | std::ios::binary);
        matwrite(fs, m_weightsInputToHidden);
        matwrite(fs, m_weightsHiddenToOutput);
        matwrite(fs, m_biasInputToHidden);
        matwrite(fs, m_biasHiddenToOutput);
        fs.close();
    }

    NeuralNetwork& deserialize(const char *path) {
        std::ifstream fs(path, std::ios::in | std::ios::binary);
        m_weightsInputToHidden = matread(fs);
        m_weightsHiddenToOutput = matread(fs);
        m_biasInputToHidden = matread(fs);
        m_biasHiddenToOutput = matread(fs);
        return *this;
    }
};

#endif //DEBOGGLER_NEURALNETWORK_H
