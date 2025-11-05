#ifndef EYETOOLKIT_H
#define EYETOOLKIT_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <tuple>

namespace eyeToolKit {

// Fit ellipse to eye landmarks
std::pair<cv::RotatedRect, std::vector<cv::Point>> 
fitEllipseToEye(const std::vector<cv::Point2f>& landmarks, 
                const std::vector<int>& eyeLandmarks, 
                int h, int w);

// Extract eye region using ellipse mask
std::pair<cv::Mat, cv::Mat> 
extractEyeRegion(const cv::Mat& frame, const cv::RotatedRect& ellipse);

// Extract pupil from eye region (old method)
cv::Mat extractPupilOld(const cv::Mat& eyeRegion, const cv::Mat& eyeMask);

// Extract pupil from eye region (method 2)
cv::Mat extractPupil2(const cv::Mat& eyeRegion, const cv::Mat& eyeMask);

// Extract pupil from eye region (method 3)
cv::Mat extractPupil3(const cv::Mat& eyeRegion, const cv::Mat& eyeMask);

// Get face orientation (yaw, pitch, roll)
std::tuple<double, double, double> 
getFaceOrientation(const std::vector<cv::Point2f>& landmarks,
                   int chinIdx, int noseIdx, 
                   int leftEyeIdx, int rightEyeIdx,
                   int leftMouthIdx, int rightMouthIdx);

// Get face size
double getFaceSize(const std::vector<cv::Point2f>& landmarks,
                   int chinIdx, int noseIdx,
                   int leftEyeIdx, int rightEyeIdx);

// Generate red circle on white background
cv::Mat genRedCircleOnWhiteBg(const cv::Size& size, 
                               int circleRadius, 
                               const cv::Point& circlePosition);

// Compute gaze physically
std::tuple<double, double, double, double>
computeGazePhysically(double rightEyeYaw, double leftEyeYaw,
                      double rightEyePitch, double leftEyePitch,
                      double faceYaw, double facePitch,
                      double faceX, double faceY, double faceZ);

} // namespace eyeToolKit

#endif // EYETOOLKIT_H

