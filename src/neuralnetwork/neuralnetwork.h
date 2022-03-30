//
// Created by Roman SAHEL on 05/02/2022.
//

#ifndef DEBOGGLER_NEURALNETWORK_H
#define DEBOGGLER_NEURALNETWORK_H

#include <iostream>
#include <fstream>
#include <random>       // std::default_random_engine

#include <opencv2/core/core.hpp>

#include "serialization.h"

template<typename UnaryFunc, typename Mat>
Mat matmap(Mat &&input, UnaryFunc func)
{
    auto outputPtr = (float *) (input.template begin<float>().ptr);
    auto outputPtrEnd = (float *) (input.template end<float>().ptr);
    for (; outputPtr != outputPtrEnd; ++outputPtr)
        *outputPtr = func(*outputPtr);
    return input;
}

#define print_mat(MAT) print(MAT, #MAT)

template<typename Mat>
auto &print(Mat &&input, const char *name)
{
    return std::cout << name << " = " << std::endl << " " << input << std::endl;
}

float sigmoid(float x)
{
    return 1.0f / (1 + std::exp(-x));
}

float dsigmoid(float sigmoid)
{
    return sigmoid * (1.0f - sigmoid);
}

struct TrainingData
{
    cv::Mat inputs;
    cv::Mat targets;

    TrainingData(const Mat &inputs, const Mat &targets) : inputs(inputs), targets(targets)
    {}
};

struct NeuralNetwork
{
    cv::Mat m_weights[2];
    cv::Mat m_bias[2];

    NeuralNetwork() = default;

    NeuralNetwork(int nbInputs, int nbHiddens, int nbOutputs)
    {
        m_weights[0] = cv::Mat::zeros(nbHiddens, nbInputs, CV_32FC1);
        cv::randu(m_weights[0], -1.0f, 1.0f);
        m_bias[0] = cv::Mat::zeros(nbHiddens, 1, CV_32FC1);
        cv::randu(m_bias[0], -1.0f, 1.0f);

        m_weights[1] = cv::Mat::zeros(nbOutputs, nbHiddens, CV_32FC1);
        cv::randu(m_weights[1], -1.0f, 1.0f);
        m_bias[1] = cv::Mat::zeros(nbOutputs, 1, CV_32FC1);
        cv::randu(m_bias[1], -1.0f, 1.0f);
    }

    [[nodiscard]] cv::Mat feed_forward_to_hiddens(const cv::Mat &inputs) const
    {
        return matmap(cv::Mat((m_weights[0] * inputs) + m_bias[0]), sigmoid);
    }

    [[nodiscard]] cv::Mat feed_forward_to_outputs(const cv::Mat &hiddens) const
    {
        return matmap(cv::Mat((m_weights[1] * hiddens) + m_bias[1]), sigmoid);
    }

    [[nodiscard]] cv::Mat feed_forward(const cv::Mat &inputs) const
    {
        return feed_forward_to_outputs(feed_forward_to_hiddens(inputs));
    }

    template<class TData>
    float train(std::vector<TData> &trainingData, int epochs, int batchSize, float learningRate = 0.05f)
    {
        static unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

        cv::Mat nabla_weights[2];
        cv::Mat nabla_bias[2];
        for (int i = 0; i < 2; ++i)
        {
            nabla_weights[i] = cv::Mat::zeros(m_weights[i].rows, m_weights[i].cols, m_weights[i].type());
            nabla_bias[i] = cv::Mat::zeros(m_bias[i].rows, m_bias[i].cols, m_bias[i].type());
        }

        float error = 0.0;
        int totalCount = 0;
        for (int i = 0; i < epochs; ++i)
        {
            std::shuffle(trainingData.begin(), trainingData.end(), std::default_random_engine(seed));

            int count = 0;
            auto batchBegin = trainingData.begin();
            while (batchBegin != trainingData.end())
            {
                auto batchEnd = std::distance(batchBegin, trainingData.end()) > batchSize ? batchBegin + batchSize : trainingData.end();
                for (;batchBegin != batchEnd; batchBegin++) {
                    error += backpropagate(*batchBegin, nabla_weights, nabla_bias, learningRate);
                    count++;
                }

                float inv_factor = 1.0f / float(count);
                for (int k = 0; k < 2; ++k)
                {
                    m_weights[k] += nabla_weights[k] * inv_factor;
                    nabla_weights[k] = 0;
                    m_bias[k] += nabla_bias[k] * inv_factor;
                    nabla_bias[k] = 0;
                }
                totalCount += count;
                count = 0;
            }

//            std::cout << "Epoch << " << i << ": " << (error / float (totalCount)) << std::endl;
        }

        return error / float (totalCount);
    }

    float backpropagate(const TrainingData &trainingData, cv::Mat nabla_weights[], cv::Mat nabla_bias[], float learningRate) const
    {
        auto hiddens = feed_forward_to_hiddens(trainingData.inputs);
        auto outputs = feed_forward_to_outputs(hiddens);

        // Outputs to Hiddens backpropagation
        auto errorOutputToHidden = trainingData.targets - outputs;
        // calculate deltas outputs->hiddens: lr * Errors * (Outputs*(1-Outputs)) * transpose(Hiddens)
        auto gradientOutputToHidden = learningRate * errorOutputToHidden.mul(matmap(outputs.clone(), dsigmoid));
        auto deltaOutputToHidden = gradientOutputToHidden * hiddens.t();
        nabla_weights[1] += deltaOutputToHidden;
        nabla_bias[1] += gradientOutputToHidden;

        // Hiddens to Inputs backpropagation
        cv::Mat errorHiddenToInput = m_weights[1].t() * errorOutputToHidden;
        // calculate deltas hiddens->inputs: lr * Errors * (Hiddens*(1-Hiddens)) * transpose(Inputs)
        auto gradientHiddenToInput = learningRate * errorHiddenToInput.mul(matmap(hiddens.clone(), dsigmoid));
        auto deltaHiddenToInput = gradientHiddenToInput * trainingData.inputs.t();
        nabla_weights[0] += deltaHiddenToInput;
        nabla_bias[0] += gradientHiddenToInput;
        return norm(errorOutputToHidden);
    }

    void serialize(const char *path) const
    {
        std::ofstream fs(path, std::ios::out | std::ios::binary);
        matwrite(fs, m_weights[0]);
        matwrite(fs, m_weights[1]);
        matwrite(fs, m_bias[0]);
        matwrite(fs, m_bias[1]);
        fs.close();
    }

    NeuralNetwork &deserialize(const char *path)
    {
        std::ifstream fs(path, std::ios::in | std::ios::binary);
        m_weights[0] = matread(fs);
        m_weights[1] = matread(fs);
        m_bias[0] = matread(fs);
        m_bias[1] = matread(fs);
        return *this;
    }
};

#endif //DEBOGGLER_NEURALNETWORK_H
