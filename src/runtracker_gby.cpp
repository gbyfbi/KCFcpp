#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "kcftracker.hpp"

#include <dirent.h>
#include <boost/program_options.hpp>

using namespace std;
using namespace cv;
namespace po = boost::program_options;

int main(int argc, char* argv[]){
    int opt;
    po::options_description desc("Allowed options");

	bool HOG = false;
	bool FIXEDWINDOW = false;
	bool MULTISCALE = false;
	bool SILENT = false;
	bool LAB = false;
//	string fileName = "images.txt";
	string groundTruth = "region.txt";
	vector<string> initRegionFilePathList;
	string imageListFilePath = "images.txt";
	string resultsPath = "output.txt";
	std::vector<string> defaultResionFilePathList(1, std::string("region.txt"));
	desc.add_options()
			("help,h", "produce help message")
//			("optimization", po::value<int>(&opt)->default_value(10), "optimization level")
			("hog,H", po::bool_switch(&HOG)->default_value(false), "if use hog feature")
			("lab,L", po::bool_switch(&LAB)->default_value(false), "if use lab feature")
			("fixed-window,F", po::bool_switch(&FIXEDWINDOW)->default_value(false), "if use fixed window")
			("multi-scale,M", po::bool_switch(&MULTISCALE)->default_value(false), "if use multiple scale")
			("silent,S", po::bool_switch(&SILENT)->default_value(false), "if be silent and do not display")
			("image-list,I", po::value<string>(&imageListFilePath)->default_value(string("images.txt")), "the input image list file path")
			("init-region,R", po::value< vector<string> >(&initRegionFilePathList), "the input initial region file path(s)")
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
		std::vector<string> tt = vm["init-region"].as<std::vector<std::string> >();
		for (int i = 0; i < tt.size(); i++) {
			cout << "\t" << tt[i] <<endl;
		}
		groundTruth = tt[0];
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

//	for(int i = 0; i < argc; i++){
//		if ( strcmp (argv[i], "hog") == 0 )
//			HOG = true;
//		if ( strcmp (argv[i], "fixed_window") == 0 )
//			FIXEDWINDOW = true;
//		if ( strcmp (argv[i], "singlescale") == 0 )
//			MULTISCALE = false;
//		if ( strcmp (argv[i], "show") == 0 )
//			SILENT = false;
//		if ( strcmp (argv[i], "lab") == 0 ){
//			LAB = true;
//			HOG = true;
//		}
//		if ( strcmp (argv[i], "gray") == 0 )
//			HOG = false;
//	}
	
	// Create KCFTracker object
	KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);

	// Frame readed
	Mat frame;

	// Tracker results
	Rect result;

	// Path to list.txt
//	ifstream listFile;
//  	listFile.open(fileName);

  	// Read groundTruth for the 1st frame
  	ifstream groundtruthFile;
  	groundtruthFile.open(groundTruth);
  	string firstLine;
  	getline(groundtruthFile, firstLine);
	groundtruthFile.close();
  	
  	istringstream ss(firstLine);

  	// Read groundTruth like a dumb
  	float x1, y1, x2, y2, x3, y3, x4, y4;
  	char ch;
	ss >> x1;
	ss >> ch;
	ss >> y1;
	ss >> ch;
	ss >> x2;
	ss >> ch;
	ss >> y2;
	ss >> ch;
	ss >> x3;
	ss >> ch;
	ss >> y3;
	ss >> ch;
	ss >> x4;
	ss >> ch;
	ss >> y4; 

	// Using min and max of X and Y for groundTruth rectangle
	float xMin =  min(x1, min(x2, min(x3, x4)));
	float yMin =  min(y1, min(y2, min(y3, y4)));
	float width = max(x1, max(x2, max(x3, x4))) - xMin;
	float height = max(y1, max(y2, max(y3, y4))) - yMin;

	
	// Read Images
	ifstream listFramesFile;
	listFramesFile.open(imageListFilePath);
	string frameName;


	// Write Results
	ofstream resultsFile;
	resultsFile.open(resultsPath);

	// Frame counter
	int nFrames = 0;


	while ( getline(listFramesFile, frameName) ){
		frameName = frameName;

		// Read each frame from the list
		printf("%s\n", frameName.c_str());
		fflush(stdout);
		frame = imread(frameName, CV_LOAD_IMAGE_COLOR);

		// First frame, give the groundTruth to the tracker
		if (nFrames == 0) {
			tracker.init( Rect(xMin, yMin, width, height), frame );
			rectangle( frame, Point( xMin, yMin ), Point( xMin+width, yMin+height), Scalar( 0, 255, 255 ), 1, 8 );
            circle( frame, Point( xMin, yMin ), 10, cv::Scalar(0, 0, 255), 5 );
			resultsFile << xMin << "," << yMin << "," << width << "," << height << endl;
		}
		// Update
		else{
			result = tracker.update(frame);
			rectangle( frame, Point( result.x, result.y ), Point( result.x+result.width, result.y+result.height), Scalar( 0, 255, 255 ), 1, 8 );
            circle( frame, Point( result.x, result.y), 10, cv::Scalar(0, 0, 255), 5 );
			resultsFile << result.x << "," << result.y << "," << result.width << "," << result.height << endl;
		}

		nFrames++;

		if (!SILENT){
			imshow("Image", frame);
			waitKey(1);
		}
	}
	resultsFile.close();

//	listFile.close();

}
