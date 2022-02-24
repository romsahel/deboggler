#include <iostream>

#include <random>       // std::default_random_engine
#include <chrono>       // std::chrono::system_clock

#include <opencv2/core/core.hpp>
#include "../commons.h"

#include "perceptron.h"
#include "neuralnetwork.h"


struct Data
{
    cv::Mat inputs;
    cv::Mat targets;
    char targetChar;
    std::string  path;
    Data(const std::string& path, int nbOutputs)
     : inputs(cv::imread(path, cv::IMREAD_GRAYSCALE))
     , targets(cv::Mat::zeros(nbOutputs, 1, CV_32FC1))
     , targetChar(std::filesystem::path(path).filename().string()[0])
     , path(path)
    {
        inputs = inputs.reshape(1, inputs.cols * inputs.rows);
        inputs.convertTo(inputs, CV_32FC1, 1.0f / 255.0f);

        int targetIndex = static_cast<int>(targetChar) - 'A';
        targets.at<float>(0, targetIndex) = 1.0f;
    }
};

size_t readAllImages(const char *path, std::vector<cv::String> &filepathes, std::vector<Data> &data, int nbOutputs) {
    filepathes.clear();
    data.clear();

    cv::glob(path, filepathes, true);
    size_t count = filepathes.size();
    data.reserve(count);
    for (size_t i = 0; i < count; i++) {
        data.push_back(Data(filepathes[i], nbOutputs));
    }
    return count;
}

int main(int argc, const char *argv[]) {
    constexpr int nbOutputs = 26;
    constexpr const char* serializationPath = "../neuralNetwork.bin";
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

    std::vector<Data> training;
    std::vector<cv::String> filepathes;
    size_t count = readAllImages("../output/*.jpg", filepathes, training, nbOutputs);
    int nbInputs = training[0].inputs.rows;
    auto neuralNetwork = std::filesystem::exists(serializationPath) ? NeuralNetwork().deserialize(serializationPath) : NeuralNetwork(nbInputs, 128, nbOutputs);
    auto test = training;
//    std::shuffle (training.begin(), training.end(), std::default_random_engine(seed));
//    auto test = std::vector<Data>(training.begin(), training.begin() + count / 10);
//    for (int i = 0; i < test.size(); ++i) {
//        training.pop_back();
//    }
//    count = training.size();

    for (int j = 0; j < 10000; ++j) {
        float cost = 0.0f;

        std::shuffle (training.begin(), training.end(), std::default_random_engine(seed));
        for (int i = 0; i < count; ++i) {
            cost += neuralNetwork.train(training[i].inputs, training[i].targets);
        }
        std::cout << "Epoch " << j << ": " << (cost/count) << std::endl;
    }

    neuralNetwork.serialize(serializationPath);

    float accuracy = 0.0f;
    count = test.size();
    for (int i = 0; i < count; ++i) {
        auto guess = neuralNetwork.feed_forward(test[i].inputs);
        int maxIndex = 0;
        for (int k = 1; k < nbOutputs; ++k) {
            if (guess.at<float>(0, k) > guess.at<float>(0, maxIndex)) {
                maxIndex = k;
            }
        }

        char guessedChar = (char) ('A' + maxIndex);
        if (guessedChar != test[i].targetChar) {
            std::cout << "Wrong " << i << "! Found " << guessedChar << " instead of " << test[i].targetChar << " (" << guess.at<float>(0, maxIndex) << ")" << " (" << test[i].path << ")" << std::endl; 
        } else {
            accuracy += guess.at<float>(0, maxIndex);
        }
    }

    std::cout << "Average accuracy: " << (int) ((accuracy / count) * 100.0f) << '%' << std::endl;
}

int xor_perceptron(int argc, const char *argv[]) {

    auto random = cv::RNG();
    constexpr int nbPoints = 300;

    constexpr int nbInputs = 2;
    constexpr int nbOutputs = 1;
    NeuralNetwork neuralNetwork(nbInputs, 2, nbOutputs);
    std::tuple<cv::Mat, cv::Mat> trainingData[] = {
            std::make_tuple((cv::Mat_<float>(nbInputs, 1) << 0.0f, 0.0f), (cv::Mat_<float>(nbOutputs, 1) << 0.0f)),
            std::make_tuple((cv::Mat_<float>(nbInputs, 1) << 0.0f, 1.0f), (cv::Mat_<float>(nbOutputs, 1) << 1.0f)),
            std::make_tuple((cv::Mat_<float>(nbInputs, 1) << 1.0f, 0.0f), (cv::Mat_<float>(nbOutputs, 1) << 1.0f)),
            std::make_tuple((cv::Mat_<float>(nbInputs, 1) << 1.0f, 1.0f), (cv::Mat_<float>(nbOutputs, 1) << 0.0f))
    };

    auto backgroundColor = cv::Scalar(127, 127, 127);
    cv::Mat canvas = cv::Mat::zeros(height, width, CV_8UC3);
    std::vector<float> inputs{0.0f, 0.0f, 0.0f};
    cvui::init("Neural network", 100);
    int trainingIndex = 0;
    bool isTraining = false;
    while (true) {
        canvas = backgroundColor;

        if (isTraining) {
            for (int j = 0; j < 1000; ++j) {
                int i = random.uniform(0, 4);
                neuralNetwork.train(get<0>(trainingData[i]), get<1>(trainingData[i]));
            }
        }

        constexpr int cellHeight = height / 4;
        constexpr int cellWidth = width / 3;
        // for each line
        auto trueColor = cv::Scalar(0, 255, 0);
        auto falseColor = cv::Scalar(255, 0, 0);
        for (int i = 0; i < 4; ++i) {
            const auto &data = get<0>(trainingData[i]);
            const auto guess = neuralNetwork.feed_forward(data).at<float>(0, 0);
            auto rect = cv::Rect(0, i * cellHeight, cellWidth, cellHeight);
            cv::rectangle(canvas, rect,
                          data.at<float>(0, 0) > 0.5f ? trueColor : falseColor, cv::FILLED);
            rect.x += rect.width;
            cv::rectangle(canvas, rect,
                          data.at<float>(0, 1) > 0.5f ? trueColor : falseColor, cv::FILLED);
            rect.x += rect.width + 25;
            cv::rectangle(canvas, rect,
                          guess * trueColor + (1 - guess) * falseColor, cv::FILLED);
        }

        if (cvui::button(canvas, 20, 20, "&Quit")) {
            break;
        }


        if (cvui::button(canvas, 20, 50, isTraining ? "Pause training" : "Resume training")) {
            isTraining = !isTraining;
        }

        cvui::update();
        cv::cvtColor(canvas, canvas, cv::COLOR_BGR2RGB);
        cv::imshow("Neural network", canvas);


    }

    return 0;
}