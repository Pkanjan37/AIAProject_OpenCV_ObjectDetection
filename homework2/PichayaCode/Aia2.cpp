//============================================================================
// Name        : Aia2.cpp
// Author      : Ronny Haensch
// Version     : 1.0
// Copyright   : -
// Description : 
//============================================================================

#include "Aia2.h"
#include <cmath>

// calculates the contour line of all objects in an image
/*
img			the input image
objList		vector of contours, each represented by a two-channel matrix
thresh		threshold used to binarize the image
k			number of applications of the erosion operator
*/
void Aia2::getContourLine(const Mat& img, vector<Mat>& objList, int thresh, int k) {
	cv::threshold(img, img, thresh, 255, CV_THRESH_BINARY);


	if (k > 0) {
		erode(img, img, getStructuringElement(MORPH_RECT, Size(k, k)));
	}
	else
		cout << "iteration k for erosion is not set" << endl;

	Mat canny_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	/// Detect edges using canny
	Canny(img, canny_output, thresh, thresh*2, 3);
	/// Find contours
	findContours(canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, Point(0, 0));

	//std::vector<std::vector<cv::Point> > contours;
	//cv::Mat contourOutput = img.clone();
	//findContours(contourOutput, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	cv::Mat contourImage(img.size(), CV_8UC3, cv::Scalar(0, 0, 0));
	cv::Scalar colors[3];
	colors[0] = cv::Scalar(255, 0, 0);
	colors[1] = cv::Scalar(0, 255, 0);
	colors[2] = cv::Scalar(0, 0, 255);
	for (size_t idx = 0; idx < contours.size(); idx++) {
		cv::drawContours(contourImage, contours, idx, colors[idx % 3]);
	}

	cv::imshow("Input Image", img);
	cvMoveWindow("Input Image", 0, 0);
	cv::imshow("Contours", contourImage);
	cvMoveWindow("Contours", 200, 0);
	cv::waitKey(0);
	for (int i = 0; i < contours.size(); i++)
	{
		vector<Point> currPts = contours.at(i);
		Mat currLine(currPts.size(), 1.0, CV_32FC2);
		for (int j = 0; j < currPts.size(); j++)
		{
			currLine.at<Vec2i>(j, 0).val[0] = currPts.at(j).x;
			currLine.at<Vec2i>(j, 0).val[1] = currPts.at(j).y;
		}
		objList.push_back(currLine);
	}

}

// calculates the (unnormalized!) fourier descriptor from a list of points
/*
contour		1xN 2-channel matrix, containing N points (x in first, y in second channel)
out		fourier descriptor (not normalized)
*/
Mat Aia2::makeFD(const Mat& contour) {
	//Convert to floating point precision
	Mat convContour;
	contour.convertTo(convContour, CV_32FC2);

	Mat fd;
	dft(contour, fd);

	return fd;


}

// normalize a given fourier descriptor
/*
fd		the given fourier descriptor
n		number of used frequencies (should be even)
out		the normalized fourier descriptor
*/
Mat Aia2::normFD(const Mat& fd, int n) {

	//plotFD(fd, "fd not normalized", 0);

	// translation invariance F0 = 0
	Mat tranFD;
	fd.copyTo(tranFD);
	Mat tmp[2];
	split(tranFD, tmp);
	tmp[0].at<float>(0, 0) = 0;
	tmp[1].at<float>(0, 0) = 0;
	merge(tmp, 2, tranFD);


	//plotFD(tranFD, "fd translation invariant", 0);
	// scale invariance
	Mat temp[2];
	split(tranFD, temp);
	double Re = temp[0].at<float>(1, 0);
	double Im = temp[1].at<float>(1, 0);
	double magF1 = sqrt(std::pow(Re, 2) + std::pow(Im, 2));
	
	//cerr << "/n Press enter to RE..." << Re;
	//cerr << "/n Press enter to RE..." << Im;
	//cerr << "/n Press enter to RE..." << magF1;
	if (magF1 != 0) {
		for (int i = 2; i < tranFD.rows; ++i) {
			temp[0].at<float>(i, 0) = temp[0].at<float>(i, 0) / magF1;
			temp[1].at<float>(i, 0) = temp[1].at<float>(i, 0) / magF1;
		}
		merge(temp, 2, tranFD);
	}
	
	
	//plotFD(tranFD, "fd translation and scale invariant", 0);


	// rotation invariance
	Mat planes[2];
	split(tranFD, planes);
	//cerr << "\n Press enter to planes..." << planes[1];
	//cerr << "\n Press enter to planes..." << planes[0];
	cv::cartToPolar(planes[0],planes[1], planes[0], planes[1]);
	//magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
	//cerr << "\n Press enter to planes22222222..." << ans;
	Mat rotateFD = planes[0];
	//cerr << "/n Press enter to rotateFD..." << planes[0];
	//plotFD(rotateFD, "fd translation, scale, and rotation invariant", 0);

	// smaller sensitivity for details

	Mat result[1];
	for (int i = 0; i < n; i++)
	{
		double currPts = planes[0].at<float>(i);
	
		result[0].push_back(currPts);
	}
	//cerr << "\n Press enter to resultXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXxxx..." << " " << result[0];
	//cerr << "\n Press enter to resultXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXxxx..." << " " << result[0].rows;
	//plotFD(rotateFD, "fd translation, scale, and rotation invariant, smaller sensitivity", 0);
	rotateFD = result[0];

	return rotateFD;
}

// plot fourier descriptor
/*
fd	the fourier descriptor to be displayed
win	the window name
dur	wait number of ms or until key is pressed
*/
void Aia2::plotFD(const Mat& fd, string win, double dur) {
	cv::Mat inverseTransform;
	cv::dft(fd, inverseTransform, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);
	cv::resize(inverseTransform, inverseTransform, cv::Size(), 5.00, 10.00);
	imshow(win, inverseTransform);
	waitKey(dur);
}

/* *****************************
GIVEN FUNCTIONS
***************************** */

// function loads input image, calls processing functions, and saves result
// in particular extracts FDs and compares them to templates
/*
img			path to query image
template1	path to template image of class 1
template2	path to template image of class 2
*/
void Aia2::run(string img, string template1, string template2) {

	// process image data base
	// load image as gray-scale, paths in argv[2] and argv[3]
	cv::Mat exC1 = cv::imread(template1, 0);
	cv::Mat exC2 = cv::imread(template2, 0);
	//	cv::cvtColor(exC1, exC1, CV_BGR2GRAY);
	//	cv::cvtColor(exC2, exC2, CV_BGR2GRAY);
	if ((!exC1.data) || (!exC2.data)) {
		cout << "ERROR: Cannot load class examples in\n" << template1 << "\n" << template2 << endl;
		cerr << "Press enter to continue..." << endl;
		cin.get();
		exit(-1);
	}

	// parameters
	// these two will be adjusted below for each image indiviudally
	int binThreshold;				// threshold for image binarization
	int numOfErosions;				// number of applications of the erosion operator
									// these two values work fine, but might be interesting for you to play around with them
	int steps = 32;					// number of dimensions of the FD
	double detThreshold = 0.01;		// threshold for detection

									// get contour line from images
	vector<Mat> contourLines1;
	vector<Mat> contourLines2;
	// TO DO !!!
	// --> Adjust threshold and number of erosion operations
	binThreshold = 140;
	numOfErosions = 5;
	getContourLine(exC1, contourLines1, binThreshold, numOfErosions);
	int mSize = 0, mc1 = 0, mc2 = 0, i = 0;
	for (vector<Mat>::iterator c = contourLines1.begin(); c != contourLines1.end(); c++, i++) {
		if (mSize<c->rows) {
			mSize = c->rows;
			mc1 = i;
		}
	}
	getContourLine(exC2, contourLines2, binThreshold, numOfErosions);
	for (vector<Mat>::iterator c = contourLines2.begin(); c != contourLines2.end(); c++, i++) {
		if (mSize<c->rows) {
			mSize = c->rows;
			mc2 = i;
		}
	}
	// calculate fourier descriptor
	Mat fd1 = makeFD(contourLines1.at(mc1));
	Mat fd2 = makeFD(contourLines2.at(mc2));

	// normalize  fourier descriptor
	Mat fd1_norm = normFD(fd1, steps);
	Mat fd2_norm = normFD(fd2, steps);

	// process query image
	// load image as gray-scale, path in argv[1]
	Mat query = imread(img, 0);
	if (!query.data) {
		cerr << "ERROR: Cannot load query image in\n" << img << endl;
		cerr << "Press enter to continue..." << endl;
		cin.get();
		exit(-1);
	}

	// get contour lines from image
	vector<Mat> contourLines;
	// TO DO !!!
	// --> Adjust threshold and number of erosion operations
	binThreshold = 100;
	numOfErosions = 3;
	getContourLine(query, contourLines, binThreshold, numOfErosions);

	cout << "\n Found " << contourLines.size() << " object candidates" << endl;

	// just to visualize classification result
	Mat result(query.rows, query.cols, CV_8UC3);
	vector<Mat> tmp;
	tmp.push_back(query);
	tmp.push_back(query);
	tmp.push_back(query);
	merge(tmp, result);
	
	// loop through all contours found
	i = 1;
	for (vector<Mat>::iterator c = contourLines.begin(); c != contourLines.end(); c++, i++) {

		cout << "Checking object candidate no " << i << " :\t";

		// color current object in yellow
		Vec3b col(0, 255, 255);
		for (int p = 0; p < c->rows; p++) {
			result.at<Vec3b>(c->at<Vec2i>(p)[1], c->at<Vec2i>(p)[0]) = col;
		}
		//showImage(result, "result", 0);

		// if fourier descriptor has too few components (too small contour), then skip it (and color it in blue)
		if (c->rows < steps) {
			cout << "Too less boundary points (" << c->rows << " instead of " << steps << ")" << endl;
			col = Vec3b(255, 0, 0);
		}
		else {
			// calculate fourier descriptor
			Mat fd = makeFD(*c);
			// normalize fourier descriptor
			Mat fd_norm = normFD(fd, steps);
			// compare fourier descriptors

			double err1 = norm(fd_norm, fd1_norm) / steps;
			double err2 = norm(fd_norm, fd2_norm) / steps;

			// if similarity is too small, then reject (and color in cyan)
			if (min(err1, err2) > detThreshold) {
				cout << "No class instance ( " << min(err1, err2) << " )" << endl;
				
				col = Vec3b(255, 255, 0);
			}
			else {
				// otherwise: assign color according to class
				if (err1 > err2) {
					col = Vec3b(0, 0, 255);
					
					cout << "Class 2 ( " << err2 << " )" << endl;
				}
				else {
					col = Vec3b(0, 255, 0);
					cout << "Class 1 ( " << err1 << " )" << endl;
				}
			}
		}
		
		// draw detection result
		for (int p = 0; p < c->rows; p++) {

			result.at<Vec3b>(c->at<Vec2i>(p)[1], c->at<Vec2i>(p)[0]) = col;
		
		}
		// for intermediate results, use the following line
		//showImage(result, "result", 0);

	}
	// save result
	imwrite("result.png", result);
	// show final result
	showImage(result, "result", 0);
}

// shows the image
/*
img	the image to be displayed
win	the window name
dur	wait number of ms or until key is pressed
*/
void Aia2::showImage(const Mat& img, string win, double dur) {

	// use copy for normalization
	Mat tempDisplay = img.clone();
	if (img.channels() == 1) normalize(img, tempDisplay, 0, 255, CV_MINMAX);
	// create window and display omage
	namedWindow(win.c_str(), CV_WINDOW_AUTOSIZE);
	imshow(win.c_str(), tempDisplay);
	// wait
	if (dur >= 0) waitKey(dur);

}

// function loads input image and calls processing function
// output is tested on "correctness" 
void Aia2::test(void) {

	test_getContourLine();
	////test_makeFD();
	//test_normFD();

}

void Aia2::test_getContourLine(void) {
	double eps = pow(10, -3);
	vector<Mat> objList;
	Mat img(100, 100, CV_8UC1, Scalar(255));
	Mat roi(img, Rect(40, 40, 20, 20));
	roi.setTo(0);
	getContourLine(img, objList, 128, 1);
	Mat fd = makeFD(objList.at(0));
	Mat nfd = normFD(fd, 32);
	if (fd.rows != objList.at(0).rows) {
		cout << "There is be a problem with Aia2::makeFD(..):" << endl;
		cout << "\tThe number of frequencies does not match the number of contour points" << endl;
		cin.get();
		exit(-1);
	}
	if (fd.channels() != 2) {
		cout << "There is be a problem with Aia2::makeFD(..):" << endl;
		cout << "\tThe fourier descriptor is supposed to be a two-channel, 1D matrix" << endl;
		cin.get();
		exit(-1);
	}
	if (nfd.channels() != 1) {
		cout << "There is be a problem with Aia2::normFD(..):" << endl;
		cout << "\tThe normalized fourier descriptor is supposed to be a one-channel, 1D matrix" << endl;
		cin.get();
		exit(-1);
	}
	if (abs(nfd.at<float>(0)) > eps) {
		cout << "There is be a problem with Aia2::normFD(..):" << endl;
		cout << "\tThe F(0)-component of the normalized fourier descriptor F is supposed to be 0" << endl;
		cin.get();
		exit(-1);
	}
	if ((abs(nfd.at<float>(1) - 1.) > eps) && (abs(nfd.at<float>(nfd.rows - 1) - 1.) > eps)) {
		cout << "There is be a problem with Aia2::normFD(..):" << endl;
		cout << "\tThe F(1)-component of the normalized fourier descriptor F is supposed to be 1" << endl;
		cout << "\tBut what if the unnormalized F(1)=0?" << endl;
		cin.get();
		exit(-1);
	}
	if (nfd.rows != 32) {
		cout << "There is be a problem with Aia2::normFD(..):" << endl;
		cout << "\tThe number of components does not match the specified number of components" << endl;
		cin.get();
		exit(-1);
	}
	Mat cline(68, 1, CV_32SC2);
	int k = 0;

	for (int i = 41; i<58; i++) cline.at<Vec2i>(k++) = Vec2i(41, i);
	for (int i = 41; i<58; i++) cline.at<Vec2i>(k++) = Vec2i(i, 58);
	for (int i = 58; i>41; i--) cline.at<Vec2i>(k++) = Vec2i(58, i);
	for (int i = 58; i>41; i--) cline.at<Vec2i>(k++) = Vec2i(i, 41);
/*		if ( sum(cline != objList.at(0)).val[0] != 0 ){
	cout << "There might be a problem with Aia2::getContourLine(..)!" << endl;
	cin.get();
	}*/

}

void Aia2::test_makeFD(void) {

	Mat cline(68, 1, CV_32SC2);
	int k = 0;
	for (int i = 41; i<58; i++) cline.at<Vec2i>(k++) = Vec2i(41, i);
	for (int i = 41; i<58; i++) cline.at<Vec2i>(k++) = Vec2i(i, 58);
	for (int i = 58; i>41; i--) cline.at<Vec2i>(k++) = Vec2i(58, i);
	for (int i = 58; i>41; i--) cline.at<Vec2i>(k++) = Vec2i(i, 41);

	Mat fd = makeFD(cline);
	if (fd.rows != cline.rows) {
		cout << "There is be a problem with Aia2::makeFD(..):" << endl;
		cout << "\tThe number of frequencies does not match the number of contour points" << endl;
		cin.get();
		exit(-1);
	}
	if (fd.channels() != 2) {
		cout << "There is be a problem with Aia2::makeFD(..):" << endl;
		cout << "\tThe fourier descriptor is supposed to be a two-channel, 1D matrix" << endl;
		cin.get();
		exit(-1);
	}
}

void Aia2::test_normFD(void) {

	double eps = pow(10, -3);

	Mat cline(68, 1, CV_32FC2);
	int k = 0;
	for (int i = 41; i<58; i++) cline.at<Vec2i>(k++) = Vec2i(41, i);
	for (int i = 41; i<58; i++) cline.at<Vec2i>(k++) = Vec2i(i, 58);
	for (int i = 58; i>41; i--) cline.at<Vec2i>(k++) = Vec2i(58, i);
	for (int i = 58; i>41; i--) cline.at<Vec2i>(k++) = Vec2i(i, 41);

	Mat fd = makeFD(cline);
	Mat nfd = normFD(fd, 32);
	if (nfd.channels() != 1) {
		cout << "There is be a problem with Aia2::normFD(..):" << endl;
		cout << "\tThe normalized fourier descriptor is supposed to be a one-channel, 1D matrix" << endl;
		cin.get();
		exit(-1);
	}
	if (abs(nfd.at<float>(0)) > eps) {
		cout << "There is be a problem with Aia2::normFD(..):" << endl;
		cout << "\tThe F(0)-component of the normalized fourier descriptor F is supposed to be 0" << endl;
		cin.get();
		exit(-1);
	}
	if ((abs(nfd.at<float>(1) - 1.) > eps) && (abs(nfd.at<float>(nfd.rows - 1) - 1.) > eps)) {
		cout << "There is be a problem with Aia2::normFD(..):" << endl;
		cout << "\tThe F(1)-component of the normalized fourier descriptor F is supposed to be 1" << endl;
		cout << "\tBut what if the unnormalized F(1)=0?" << endl;
		cin.get();
		exit(-1);
	}
	if (nfd.rows != 32) {
		cout << "There is be a problem with Aia2::normFD(..):" << endl;
		cout << "\tThe number of components does not match the specified number of components" << endl;
		cin.get();
		exit(-1);
	}
}
