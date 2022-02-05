#include <iostream>
#include <opencv2/core/core.hpp>
#include "../commons.h"

#include "perceptron.h"
#include "neuralnetwork.h"


struct CanvasPoint {
    cv::Point2f point;
    int target;
    float bias = 1.0f;

    CanvasPoint(float x, float y) : point(x, y), target(y >= line(x) ? 1 : -1) {
    }

    std::vector<float> &inputs(std::vector<float> &inputs) {
        inputs[0] = point.x;
        inputs[1] = point.y;
        inputs[2] = bias;
        return inputs;
    }

    void draw(cv::Mat &canvas, int guess) const {
        const static auto white = cv::Scalar(255, 255, 255);
        const static auto black = cv::Scalar(0, 0, 0);
        const static auto green = cv::Scalar(0, 255, 0);
        const static auto red = cv::Scalar(255, 0, 0);
        constexpr int radius = 10;
        auto color = target == 1 ? white : target == -1 ? black : red;
        cv::circle(canvas, remap(point), radius, color, cv::FILLED);
        cv::circle(canvas, remap(point), radius / 3, target == guess ? green : red, cv::FILLED);
    }
};

int main(int argc, const char *argv[]) {

    if (argc + 1 > 0) {
        auto random = cv::RNG();
        constexpr int nbPoints = 300;

        constexpr int nbInputs = 2;
        constexpr int nbOutputs = 2;
        NeuralNetwork neuralNetwork(nbInputs, 2, nbOutputs);
        cv::Mat testinput = (cv::Mat_<float>(nbInputs, 1) << 1.0f, 0.0f);
        cv::Mat testoutput = (cv::Mat_<float>(nbOutputs, 1) << 1.0f, 0.0f);
        neuralNetwork.train(testinput, testoutput);

        Perceptron perceptron{3};

        std::vector<CanvasPoint> points;
        for (int i = 0; i < nbPoints; ++i) {
            points.push_back(CanvasPoint{random.uniform(-1.0f, 1.0f), random.uniform(-1.0f, 1.0f)});
        }

        auto backgroundColor = cv::Scalar(127, 127, 127);
        cv::Mat canvas = cv::Mat::zeros(height, width, CV_8UC3);
        std::vector<float> inputs{0.0f, 0.0f, 0.0f};
        cvui::init("Perceptron", 100);
        int trainingIndex = 0;
        while (true) {
            canvas = backgroundColor;

            for (int i = 0; i < points.size(); ++i) {
                int guess = perceptron.guess(points[i].inputs(inputs));
                points[i].draw(canvas, guess);
            }

            auto pt0 = remap(cv::Point2f(-1, line(-1)));
            auto pt1 = remap(cv::Point2f(1, line(1)));
            cv::line(canvas, pt0, pt1, cv::Scalar(255, 255, 255));

            perceptron.draw(canvas);

            if (cvui::button(canvas, 20, 20, "&Quit")) {
                break;
            }


            if (cvui::button(canvas, 20, 50, "Train")) {
                for (int i = 0; i < points.size(); ++i) {
                    perceptron.train(points[i].inputs(inputs), points[i].target);
                }
            }

            cvui::update();
            cv::cvtColor(canvas, canvas, cv::COLOR_BGR2RGB);
            cv::imshow("Perceptron", canvas);


        }

        return 0;
    }

    return 0;
}