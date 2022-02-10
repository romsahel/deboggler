#include <iostream>
#include <opencv2/highgui/highgui.hpp>

#include "commons.h"
#include "FindWhiteBlobs.h"
#include "ExtractDies.h"
#include "Assembly.h"
#include "MergeWhiteBlobs.h"


int main(int argc, const char *argv[]) {
    Assembly assembly;
    assembly.showMask = false;
    assembly.push_back(new FindWhiteBlobs());
    assembly.push_back(new MergeWhiteBlobs());
    assembly.push_back(new ExtractDies(assembly));
    assembly.init();

//    for (int i = 0; i < assembly.sources.size(); ++i) {
//        assembly.load(cv::imread(assembly.sources[i]));
//        assembly.sourceIndex = i;
//        assembly.update();
//    }
    assembly.maxStep = 1;

    while (true) {
        assembly.update();
        if (assembly.draw()) {
            break;
        }
        cvui::update();
        assembly.show();
    }

    return 0;
}