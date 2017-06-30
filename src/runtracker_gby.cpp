#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "kcftracker.hpp"

#include <dirent.h>
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>

using namespace std;
using namespace cv;
namespace po = boost::program_options;
class FourPoints {
public:
	float x1, y1, x2, y2, x3, y3, x4, y4;
	float xMin, yMin, width, height;
	FourPoints(std::string line) {
		boost::algorithm::trim_if(line, boost::is_any_of("\n \t,;:"));
		std::vector<std::string> str_component_vec;
		boost::algorithm::split(str_component_vec, line, boost::is_any_of("\t ,;."));
		if (str_component_vec.size() != 8) {
			cerr << "region line must contain 8 integers!" <<endl;
			throw "error";
		}
		x1 = std::stof(str_component_vec[0]);
		y1 = std::stof(str_component_vec[1]);
		x2 = std::stof(str_component_vec[2]);
		y2 = std::stof(str_component_vec[3]);
		x3 = std::stof(str_component_vec[4]);
		y3 = std::stof(str_component_vec[5]);
		x4 = std::stof(str_component_vec[6]);
		y4 = std::stof(str_component_vec[7]);
		xMin =  min(x1, min(x2, min(x3, x4)));
		yMin =  min(y1, min(y2, min(y3, y4)));
		width = max(x1, max(x2, max(x3, x4))) - xMin;
		height = max(y1, max(y2, max(y3, y4))) - yMin;
	}
};
class InitRegionSet {
public:
	std::vector<FourPoints> regionDescVec;
    InitRegionSet(std::vector<std::string> initRegionDescFilePathVec) {
		for (int i = 0; i < initRegionDescFilePathVec.size(); i++) {
			std::string regionDescFilePath = initRegionDescFilePathVec[i];
			ifstream groundtruthFile;
			groundtruthFile.open(regionDescFilePath);
			std::string regionLine;
			getline(groundtruthFile, regionLine);
			boost::algorithm::trim_if(regionLine, boost::is_any_of("\n \t"));
			FourPoints oneRegion(regionLine);
			regionDescVec.push_back(oneRegion);
		}
	}
};
int main(int argc, char* argv[]){
    int opt;
    po::options_description desc("Allowed options");

	bool HOG = false;
	bool FIXEDWINDOW = false;
	bool MULTISCALE = false;
	bool SILENT = false;
	bool LAB = false;
	vector<string> initRegionFilePathVec;
	string imageListFilePath = "images.txt";
	string resultsPath = "output.txt";
	desc.add_options()
			("help,h", "produce help message")
//			("optimization", po::value<int>(&opt)->default_value(10), "optimization level")
			("hog,H", po::bool_switch(&HOG)->default_value(false), "if use hog feature")
			("lab,L", po::bool_switch(&LAB)->default_value(false), "if use lab feature")
			("fixed-window,F", po::bool_switch(&FIXEDWINDOW)->default_value(false), "if use fixed window")
			("multi-scale,M", po::bool_switch(&MULTISCALE)->default_value(false), "if use multiple scale")
			("silent,S", po::bool_switch(&SILENT)->default_value(false), "if be silent and do not display")
			("image-list,I", po::value<string>(&imageListFilePath)->default_value(string("images.txt")), "the input image list file path")
			("init-region,R", po::value< vector<string> >(&initRegionFilePathVec), "the input initial region file path(s)")
			("output,O", po::value<string>(&resultsPath), "the output tracking result path")
//			("input-file", po::value< vector<string> >(), "input file")
			;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    bool argOk = true;
	if (vm.count("help")) {
		cout << desc << "\n";
		return 1;
	}
    if (HOG) {
		cout << "HOG feature ON." << endl;
	} else {
		cout << "HOG feature OFF." << endl;
	}
	if (LAB) {
		cout << "LAB feature ON." << endl;
	} else {
		cout << "LAB feature OFF." << endl;
	}
	if (FIXEDWINDOW) {
		cout << "fixed-window feature ON." << endl;
	} else {
		cout << "fixed-window feature OFF." << endl;
	}
	if (MULTISCALE) {
		cout << "multi-scale feature ON." << endl;
	} else {
		cout << "multi-scale feature OFF." << endl;
	}
	if (SILENT) {
		cout << "silent feature ON." << endl;
	} else {
		cout << "silent feature OFF." << endl;
	}
	if (vm.count("image-list")) {
		cout << "--image-list ON with: " << imageListFilePath << endl;
	} else {
		cout << "--image-list OFF with default value: " << imageListFilePath << endl;
	}
	if (vm.count("init-region"))
	{
		cout << "--init-region ON with:" << endl;
		initRegionFilePathVec = vm["init-region"].as<std::vector<std::string> >();
		for (int i = 0; i < initRegionFilePathVec.size(); i++) {
			cout << "\t" << initRegionFilePathVec[i] <<endl;
		}
	} else {
        fflush(stdout);
		cerr << "--init-region is missing!(at least one should be specified.)" << endl;
		fflush(stderr);
		argOk = false;
	}
	if (vm.count("output")) {
		cout << "--output ON with: " << resultsPath << endl;
	} else {
		fflush(stdout);
		cerr << "--output REQUIRED!" << endl;
		fflush(stderr);
		argOk = false;
	}
    if (!argOk) {
		fflush(stdout);
		fflush(stderr);
		cout << endl << desc << endl;
		return -1;
	}

	InitRegionSet initRegionSet(initRegionFilePathVec);
	// Create KCFTracker object
	std::vector<KCFTracker> trackerVec;
    for (int i = 0; i < initRegionFilePathVec.size(); i++) {
		KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);
		trackerVec.push_back(tracker);
	}

	// Frame readed
	Mat frame;
	// Tracker results
	Rect result;
	// Read Images
	ifstream listFramesFile;
	listFramesFile.open(imageListFilePath);
	string frameName;


	// Write Results
//	ofstream resultsFile;
//	resultsFile.open(resultsPath);

	// Frame counter
	int nFrames = 0;
	while ( getline(listFramesFile, frameName) ){
        cout << frameName << endl << flush;
        frame = imread(frameName, CV_LOAD_IMAGE_COLOR);
        cv::Mat frameToDisplay;
        frame.copyTo(frameToDisplay);
        cv::Mat frameToWork;
        frame.copyTo(frameToWork);
        for (int i = 0; i < initRegionFilePathVec.size(); i++) {
			KCFTracker & tracker = trackerVec[i];
			const float xMin = initRegionSet.regionDescVec[i].xMin;
			const float yMin = initRegionSet.regionDescVec[i].yMin;
			const float width = initRegionSet.regionDescVec[i].width;
			const float height = initRegionSet.regionDescVec[i].height;
			// First frame, give the groundTruth to the tracker
			if (nFrames == 0) {
				tracker.init( Rect(xMin, yMin, width, height), frameToWork );
				rectangle( frameToDisplay, Point( xMin, yMin ), Point( xMin+width, yMin+height), Scalar( 0, 255, 255 ), 1, 8 );
//				circle( frame, Point( xMin, yMin ), 10, cv::Scalar(0, 0, 255), 5 );
//				resultsFile << xMin << "," << yMin << "," << width << "," << height << endl;
			}
				// Update
			else{
				result = tracker.update(frameToWork);
				rectangle( frameToDisplay, Point( result.x, result.y ), Point( result.x+result.width, result.y+result.height), Scalar( 0, 255, 255 ), 1, 8 );
//				circle( frame, Point( result.x, result.y), 10, cv::Scalar(0, 0, 255), 5 );
//				resultsFile << result.x << "," << result.y << "," << result.width << "," << result.height << endl;
			}
		}
        if (!SILENT){
            imshow("Image", frameToDisplay);
            waitKey(1);
        }
        nFrames++;
	}
//	resultsFile.close();
}
