#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/opencv.hpp>
#include <string>
#include <algorithm>


using namespace cv;
using namespace std;

template <typename T>
Mat cal_H(vector<Point2f> srcPoints, vector<Point2f> dstPoints, int number_of_points);
Mat gaussianfilter(const Mat input, int n, float sigmaT, float sigmaS);
bool point_comparator_1(Point2f &a, Point2f &b);
bool point_comparator_2(Point2f &a, Point2f &b);
bool compare_area(vector<Point> contour1, vector<Point> contour2);
Mat adaptive_thres(const Mat input, int n, float b);
Mat warping(Mat input, Mat T);
float cubic_calculation(float A, float B, float C, float D, float t);

/* @function main */
int main(int argc, char** argv)
{
    string filename;
    cout << "Enter the name of the image file you want to scan: \n";
    cin >> filename;
    
    Mat input = imread(filename);
    Mat input_gray;
    
    //check for validation
    if (!input.data) {
        printf("Could not open\n");
        return -1;
    }
    
    cout << "Processing the image...\n";
    
    resize(input, input, Size(round(input.cols*0.3), round(input.rows*0.3)));
    
    namedWindow("Original", WINDOW_AUTOSIZE);
    imshow("Original", input);
    
    /* 1. Edge Detection */
    
    // 1. Conversion to Grayscale
    cvtColor(input, input_gray, COLOR_RGB2GRAY);
        
    // 2. Apply Gaussian Blur
    Mat blurred = gaussianfilter(input_gray, 1, 2, 2);
    
    // 3. Apply Canny Edge Detector
    Mat canny;
    Canny(blurred, canny, 20, 10);
    
    /* 2. Document Contour Detection */
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(canny, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
    
    Mat temp1;
    input.copyTo(temp1);
    drawContours(temp1, contours, -1, Scalar(0, 255, 0), 2, 8, noArray(), 2, Point());
    namedWindow("contourrrr", WINDOW_AUTOSIZE);
    imshow("contourrrr", temp1);
    
    
    // 4. sort the contours according to the contour area
    sort(contours.begin(), contours.end(), compare_area);
    
    
    // 5. loop over the contours to find the biggest contour with exactly 4 points
    vector<vector<Point>> points;
    for (vector<Point> contour : contours) {
        vector<Point> temp;
        // 5-1. Calculate the perimeter of the contour
        double perimeter = arcLength(contour, true);
        // 5-2. Using the perimeter, approximate the polygon of the contour
        approxPolyDP(contour, temp, 0.02 * perimeter, true);
        
        // 5-3. Let's see if the approximated polygon has exactly 4 points
        // which means this is a rectangle
        if (temp.size() == 4) {
            points.push_back(temp);
            // 5-4. Break the loop!
            break;
        }
    }
    
    
    Mat contour;
    input.copyTo(contour); //The set of 4 corner points of the rectangle
    vector<Point2f> points2f;
    vector<Point> temp = points.at(0);
    for (Point2f p : temp) {
        points2f.push_back(Point2f(p));
    }
    // 6. Draw the contour, which is the document boundary
    drawContours(contour, points, -1, Scalar(0, 0, 255), 2, 8, noArray(), 2, Point());
    
    namedWindow("Contours", WINDOW_AUTOSIZE);
    imshow("Contours", contour);
    
    
    /* 3. Homograpy transformation */
    // 3-1. Order the points in a particular rule
    // bottom-left, top-left, top-right, bottom-right
    sort(points2f.begin(), points2f.end(), point_comparator_1);
    vector<Point2f> ordered;
    Point2f tl = points2f.at(0);
    Point2f br = points2f.at(3);
    sort(points2f.begin(), points2f.end(), point_comparator_2);
    
    
    Point2f tr = points2f.at(0);
    Point2f bl = points2f.at(3);
    
    ordered.push_back(bl);
    ordered.push_back(tl);
    ordered.push_back(tr);
    ordered.push_back(br);
    
    
    int width1 = sqrt(pow((br.x - bl.x), 2) + pow((br.y - bl.y), 2));
    int width2 = sqrt(pow((tr.x - tl.x), 2) + pow((tr.y - tl.y), 2));
    int maxW = max(width1, width2);
    
    int height1 = sqrt(pow((tr.x - br.x), 2) + pow((tr.y - br.y), 2));
    int height2 = sqrt(pow((tl.x - bl.x), 2) + pow((tl.y - bl.y), 2));
    int maxH = max(height1, height2);
    
    // 3-2. Order the window frame corners in the same order above
    vector<Point2f> frame_points;
    frame_points.push_back(Point2f(0, input.rows - 1));
    frame_points.push_back(Point2f(0, 0));
    frame_points.push_back(Point2f(input.cols - 1, 0));
    frame_points.push_back(Point2f(input.cols - 1, input.rows -1 ));
    
    
    // 3-3. Finally calculate the perspective transformation matrix using the points
    Mat transformation = cal_H<double>(ordered, frame_points, 4);
    
    // 3-4. Apply the perspective transform
    Mat output;
    //This function can perform interpolation
    output = warping(input_gray, transformation);
    
    //imwrite("./output/output.png", output);
    //namedWindow("Output", WINDOW_AUTOSIZE);
    //imshow("Output", output);
    
    Mat output_gray;
    // When the input image shape and document shape does not match
    if (input.rows > input.cols && maxW > maxH) {
        resize(output, output_gray, Size(input.rows, input.cols));
    }
    else {
      output.copyTo(output_gray);
    }

    
    /* 4. Adaptive Thresholding for binary image support */
    Mat output_binary = adaptive_thres(output_gray, 5, 0.9);
    
    
    /* 5. Get User Input and Save the Output Image */
    cout << "Enter 1 to save the image in grayscale" << endl;
    cout << "Enter 2 to save the image in binary image" << endl;
    int value;
    cin >> value;
    
    if (value == 1) {
        imwrite("./output/output_gray.png", output_gray);
        namedWindow("Output_gray", WINDOW_AUTOSIZE);
        imshow("Output_gray", output_gray);
    }
    else if (value == 2) {
        imwrite("./output/output_binary.png", output_binary);
        namedWindow("Output_binary", WINDOW_AUTOSIZE);
        imshow("Output_binary", output_binary);
    }
    else {
        cout << "Wrong input. Exiting the program..." << endl;
        return 0;
    }
    cout << "Image succesfully saved. Exiting the program..." << endl;
    
    waitKey(0);
    return 0;
}

/* Compare Function according to the Contour Area */
bool compare_area(vector<Point> contour1, vector<Point> contour2) {
    double one = fabs(contourArea(contour1));
    double two = fabs(contourArea(contour2));
    return one > two;
}

bool point_comparator_1(Point2f &a, Point2f &b) {
    int a_sum = a.x + a.y;
    int b_sum = b.x + b.y;
    
    return a_sum < b_sum;
}

bool point_comparator_2(Point2f &a, Point2f &b) {
    int a_diff = a.y - a.x;
    int b_diff = b.y - b.x;
    
    return a_diff < b_diff;
}

/* Implementation for substitution of warpPerspective() in OpenCV */
Mat warping(Mat input, Mat T) {
    Mat result;
    Size size = input.size();
    resize(input, result, size);
    
    T = T.inv();
    
    double x, x1, x2, y, y1, y2, z, mu, lambda;
    for (double i = 0; i < result.rows; i++) {
        for (double j = 0; j < result.cols; j++) {
            z = (T.at<double>(2, 0) * j + T.at<double>(2, 1) * i + T.at<double>(2, 2));
            y = ((T.at<double>(0, 0) * j + T.at<double>(0, 1) * i + T.at<double>(0, 2)) / z);
            x = ((T.at<double>(1, 0) * j + T.at<double>(1, 1) * i + T.at<double>(1, 2)) / z);
            
            y1 = floor(y);
            y2 = ceil(y);
            x1 = floor(x);
            x2 = ceil(x);
            
            mu = y - y1;
            lambda = x - x1;
            
            if ((y >= 0) && (x <= (result.rows - 1)) && (x >= 0) && (y <= (result.cols - 1))) {
                
                double step_1 = mu* input.at<uchar>(x1, y2) + (1.0 - mu)*input.at<uchar>(x1, y1);
                double step_2 = mu*input.at<uchar>(x2, y2) + (1.0 - mu)*input.at<uchar>(x2, y1);
                double step_3 = lambda * step_2 + (1.0 - lambda)*step_1;
                
                result.at<uchar>(i, j) = step_3;
            }
        }
    }
    return result;
}

/* Calculate Homography Matrix using SVD::compute */
template <typename T>
Mat cal_H(vector<Point2f> srcPoints, vector<Point2f> dstPoints, int number_of_points) {
    
    Mat M(2 * number_of_points, 9, CV_64F, Scalar(0));
    
    // initialize matrix
    for (int i = 0; i < number_of_points; i++) {
        Point2f pt1 = srcPoints[i];
        Point2f pt2 = dstPoints[i];
        M.at<T>(2 * i, 0) = pt1.x;
        M.at<T>(2 * i, 1) = pt1.y;
        M.at<T>(2 * i, 2) = 1;
        M.at<T>(2 * i, 3) = 0;
        M.at<T>(2 * i, 4) = 0;
        M.at<T>(2 * i, 5) = 0;
        M.at<T>(2 * i, 6) = -pt2.x * pt1.x;
        M.at<T>(2 * i, 7) = -pt2.x * pt1.y;
        M.at<T>(2 * i, 8) = -pt2.x;
        M.at<T>(2 * i + 1, 0) = 0;
        M.at<T>(2 * i + 1, 1) = 0;
        M.at<T>(2 * i + 1, 2) = 0;
        M.at<T>(2 * i + 1, 3) = pt1.x;
        M.at<T>(2 * i + 1, 4) = pt1.y;
        M.at<T>(2 * i + 1, 5) = 1;
        M.at<T>(2 * i + 1, 6) = -pt1.x * pt2.y;
        M.at<T>(2 * i + 1, 7) = -pt1.y * pt2.y;
        M.at<T>(2 * i + 1, 8) = -pt2.y;
    }
    Mat w, u, vt;
    SVD::compute(M, w, u, vt, SVD::FULL_UV);
    Mat v = vt.t();
    
    Mat temp = v.col(v.cols - 1);
    Mat H(3, 3, CV_64F);
    H.at<T>(0, 0) = temp.at<T>(0, 0);
    H.at<T>(0, 1) = temp.at<T>(1, 0);
    H.at<T>(0, 2) = temp.at<T>(2, 0);
    H.at<T>(1, 0) = temp.at<T>(3, 0);
    H.at<T>(1, 1) = temp.at<T>(4, 0);
    H.at<T>(1, 2) = temp.at<T>(5, 0);
    H.at<T>(2, 0) = temp.at<T>(6, 0);
    H.at<T>(2, 1) = temp.at<T>(7, 0);
    H.at<T>(2, 2) = temp.at<T>(8, 0);
    return H;
}

/* Gaussian filter */
Mat gaussianfilter(const Mat input, int n, float sigmaT, float sigmaS) {
    
    int row = input.rows;
    int col = input.cols;
    int kernel_size = (2 * n + 1);
    float kernel_s[3];
    float kernel_t[3];
    float denom_s = 0.0;
    float denom_t = 0.0;
    float kernelvalue;
    
    //Initializing kernel matrices
    for (int a = -n; a <= n; a++) { // denominator
        float value1 = exp(-(pow(a, 2) / (2 * pow(sigmaS, 2))));
        float value2 = exp(-(pow(a, 2)) / (2 * pow(sigmaT, 2)));
        kernel_s[a + n] = value1;
        kernel_t[a + n] = value2;
        denom_s += value1;
        denom_t += value2;
    }
    
    for (int a = -n; a <= n; a++) {  // numerator
        kernel_s[a + n] /= denom_s;
        kernel_t[a + n] /= denom_t;
    }
    
    Mat output = Mat::zeros(row, col, input.type());
    
    
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            float sum1 = 0.0;
            float sum2 = 0.0;
            for (int a = -n; a <= n; a++) {
                for (int b = -n; b <= n; b++) {
                    
                    /* Gaussian filter with "adjustkernel" process:*/
                    if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) {
                        kernelvalue = kernel_s[a + n] * kernel_t[b + n];
                        sum1 += (float)(input.at<uchar>(i + a, j + b)) * kernelvalue;
                        sum2 += kernelvalue;
                    }
                }
            }
            output.at<uchar>(i, j) = (uchar)(sum1 / sum2);
        }
    }
    return output;
}

/* Adaptive Thresholding */
Mat adaptive_thres(const Mat input, int n, float bnumber) {
    
    Mat kernel;
    
    int row = input.rows;
    int col = input.cols;
    int kernel_size = (2 * n + 1);
    
    kernel = Mat::ones(kernel_size, kernel_size, CV_32F) / (float)(kernel_size * kernel_size);
    float kernelvalue = kernel.at<float>(0, 0);
    
    Mat output = Mat::zeros(row, col, input.type());
    
    for (int i = 0; i < row; i++) { //for each pixel in the output
        for (int j = 0; j < col; j++) {
            
            float sum1 = 0.0;
            for (int a = -n; a <= n; a++) {
                for (int b = -n; b <= n; b++) {
                    /* filter with Zero-paddle boundary process*/
                    if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) { //not border pixel
                        sum1 += kernelvalue*(float)(input.at<uchar>(i + a, j + b));
                    }
                }
            }
            float temp = bnumber*(uchar)sum1;
            
            if (input.at<uchar>(i, j) > temp) output.at<uchar>(i, j) = 255;
            else output.at<uchar>(i, j) = 0;
        }
    }
    return output;
}
