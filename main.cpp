#include <opencv2/viz.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>

using namespace cv;
using namespace std;
using namespace viz;

const float calibrationSquareDimension = 0.015f; //meters
const float arucoSquareDimension = 0.049f;       //meters
const Size chessboardDimension = Size(9, 7);     //meters

void creaeteArucoMarkers()
{
    Mat outputMarker;

    Ptr<aruco::Dictionary> markerDictionary = aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME::DICT_4X4_50);

    for (int i = 0; i < 50; ++i)
    {
        aruco::drawMarker(markerDictionary, i, 200, outputMarker, 1);
        ostringstream convert;
        string imageName = "4x4Marker_";
        convert << imageName << i << ".jpg";
        imwrite(convert.str(), outputMarker);
    }
}

void createKnownBoardPosition(Size boardSize, float squareEdgeLength, vector<Point3f> &corners)
{
    for (int i = 0; i < boardSize.height; ++i)
    {
        for (int j = 0; j < boardSize.width; ++j)
        {
            corners.push_back(Point3f(j * squareEdgeLength, i * squareEdgeLength, 0.0f));
        }
    }
}

void getChessboardCorners(vector<Mat> images, vector<vector<Point2f>> &allFoundCorners, bool showResults = false)
{
    for (vector<Mat>::iterator iter = images.begin(); iter != images.end(); ++iter)
    {
        vector<Point2f> pointBuf;

        bool found = findChessboardCorners(*iter, Size(9, 7), pointBuf, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);

        if (found)
        {
            allFoundCorners.push_back(pointBuf);
        }

        if (showResults)
        {
            drawChessboardCorners(*iter, Size(9, 7), pointBuf, found);
            cv::imshow("imshow", *iter);
            waitKey(1);
        }
    }
}

void cameraCalibration(vector<Mat> calibrationImages, Size boardSize, float squareEdgeLength, Mat &cameraMatrix, Mat &distanceCoefficients)
{
    vector<vector<Point2f>> checkerboardImageSpacePoints;
    getChessboardCorners(calibrationImages, checkerboardImageSpacePoints, false);

    vector<vector<Point3f>> worldSapceCornerPoints(1);

    createKnownBoardPosition(boardSize, squareEdgeLength, worldSapceCornerPoints[0]);
    worldSapceCornerPoints.resize(checkerboardImageSpacePoints.size(), worldSapceCornerPoints[0]);

    vector<Mat> rVectors, tVectors;
    distanceCoefficients = Mat::zeros(8, 1, CV_64F);

    calibrateCamera(worldSapceCornerPoints, checkerboardImageSpacePoints, boardSize, cameraMatrix, distanceCoefficients, rVectors, tVectors);
}

void drawCubeWireframe(cv::InputOutputArray image, cv::InputArray cameraMatrix, cv::InputArray distCoeffs, cv::InputArray rvec, cv::InputArray tvec, float l)
{

    CV_Assert(image.getMat().total() != 0 && (image.getMat().channels() == 1 || image.getMat().channels() == 3));
    CV_Assert(l > 0);
    float half_l = l / 2.0;

    // project cube points
    std::vector<cv::Point3f> axisPoints;
    axisPoints.push_back(cv::Point3f(half_l, half_l, l));
    axisPoints.push_back(cv::Point3f(half_l, -half_l, l));
    axisPoints.push_back(cv::Point3f(-half_l, -half_l, l));
    axisPoints.push_back(cv::Point3f(-half_l, half_l, l));
    axisPoints.push_back(cv::Point3f(half_l, half_l, 0));
    axisPoints.push_back(cv::Point3f(half_l, -half_l, 0));
    axisPoints.push_back(cv::Point3f(-half_l, -half_l, 0));
    axisPoints.push_back(cv::Point3f(-half_l, half_l, 0));

    std::vector<cv::Point2f> imagePoints;
    projectPoints(axisPoints, rvec, tvec, cameraMatrix, distCoeffs, imagePoints);

    // draw cube edges lines
    cv::line(image, imagePoints[0], imagePoints[1], cv::Scalar(255, 0, 0), 3);
    cv::line(image, imagePoints[0], imagePoints[3], cv::Scalar(255, 0, 0), 3);
    cv::line(image, imagePoints[0], imagePoints[4], cv::Scalar(255, 0, 0), 3);
    cv::line(image, imagePoints[1], imagePoints[2], cv::Scalar(255, 0, 0), 3);
    cv::line(image, imagePoints[1], imagePoints[5], cv::Scalar(255, 0, 0), 3);
    cv::line(image, imagePoints[2], imagePoints[3], cv::Scalar(255, 0, 0), 3);
    cv::line(image, imagePoints[2], imagePoints[6], cv::Scalar(255, 0, 0), 3);
    cv::line(image, imagePoints[3], imagePoints[7], cv::Scalar(255, 0, 0), 3);
    cv::line(image, imagePoints[4], imagePoints[5], cv::Scalar(255, 0, 0), 3);
    cv::line(image, imagePoints[4], imagePoints[7], cv::Scalar(255, 0, 0), 3);
    cv::line(image, imagePoints[5], imagePoints[6], cv::Scalar(255, 0, 0), 3);
    cv::line(image, imagePoints[6], imagePoints[7], cv::Scalar(255, 0, 0), 3);
}

int startWebcamMonitoring(const Mat &cameraMatrix, const Mat &distanceCoeffiecients)
{
    Mat frame;

    vector<int> markerIds;
    vector<vector<Point2f>> markerCorners, rejectedCandicates;
    aruco::DetectorParameters parameters;

    Ptr<aruco::Dictionary> markerDictionary = aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME::DICT_4X4_50);

    VideoCapture vid(0);

    if (!vid.isOpened())
    {
        return -1;
    }

    vector<Vec3d> rvecs, tvecs;

    while (true)
    {
        if (!vid.read(frame))
        {
            break;
        }

        aruco::detectMarkers(frame, markerDictionary, markerCorners, markerIds);
        aruco::estimatePoseSingleMarkers(markerCorners, arucoSquareDimension, cameraMatrix, distanceCoeffiecients, rvecs, tvecs);

        for (int i = 0; i < markerIds.size(); ++i)
        {

            aruco::drawAxis(frame, cameraMatrix, distanceCoeffiecients, rvecs[i], tvecs[i], 0.025f);
            //drawCubeWireframe(frame, cameraMatrix, distanceCoeffiecients, rvecs[i], tvecs[i],arucoSquareDimension);
            Mat R = Mat::zeros(3, 3, CV_64F);

            Rodrigues(rvecs[i], R);

            double sy = sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) + R.at<double>(1, 0) * R.at<double>(1, 0));
            double x, y, z;
            if (!(sy < 1e-6))
            {
                x = atan2(R.at<double>(2, 1), R.at<double>(2, 2));
                y = atan2(-R.at<double>(2, 0), sy);
                z = atan2(R.at<double>(1, 0), R.at<double>(0, 0));
            }
            else
            {
                x = atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
                y = atan2(-R.at<double>(2, 0), sy);
                z = 0;
            }

            double rx = x * 180.0 / CV_PI;
            double ry = y * 180.0 / CV_PI;
            double rz = z * 180.0 / CV_PI;

            double distance = ((tvecs[i][2] + 0.02) * 0.00846) * 100;

            cout << ry << "|" << distance << endl;
        }

        cv::imshow("webcam", frame);

        char character = cv::waitKey(1);

        if (character == 27)
        {
            break;
        }
    }

    return 1;
}

bool saveCameraCalibaration(string name, Mat cameraMatrix, Mat distanceCoefficients)
{
    ofstream outStream(name);

    if (outStream)
    {
        uint16_t rows = cameraMatrix.rows;
        uint16_t colums = cameraMatrix.cols;

        outStream << rows << endl;
        outStream << colums << endl;

        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < colums; ++j)
            {
                double value = cameraMatrix.at<double>(i, j);
                outStream << value << endl;
            }
        }

        rows = distanceCoefficients.rows;
        colums = distanceCoefficients.cols;

        outStream << rows << endl;
        outStream << colums << endl;

        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < colums; ++j)
            {
                double value = distanceCoefficients.at<double>(i, j);
                outStream << value << endl;
            }
        }

        outStream.close();
        return true;
    }

    return false;
}

bool loadCameraCalibration(string name, Mat &cameraMatrix, Mat &distanceCoefficients)
{
    ifstream inStream(name);

    if (inStream)
    {
        uint16_t rows;
        uint16_t colums;

        inStream >> rows;
        inStream >> colums;

        cameraMatrix = Mat::zeros(Size(colums, rows), CV_64F);

        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < colums; ++j)
            {
                double read = 0.0f;
                inStream >> read;
                cameraMatrix.at<double>(i, j) = read;

                cout << cameraMatrix.at<double>(i, j) << "\n";
            }
        }

        inStream >> rows;
        inStream >> colums;

        distanceCoefficients = Mat::zeros(Size(colums, rows), CV_64F);

        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < colums; ++j)
            {
                double read = 0.0f;
                inStream >> read;
                distanceCoefficients.at<double>(i, j) = read;

                cout << distanceCoefficients.at<double>(i, j) << "\n";
            }
        }

        inStream.close();
        return true;
    }

    return false;
}

void AddCamera(viz::Viz3d window, String CamName, Point3d shiftP, Vec3f rotateP, bool enable)
{
    double x = rotateP[0];
    double y = rotateP[1];
    double z = rotateP[2];

    Point3d pf(cos(y) * cos(z), cos(y) * sin(z), -sin(y));
    y += CV_PI / 2.0;
    Point3d pd(cos(y) * cos(z), cos(y) * sin(z), -sin(y));

    Affine3f cam_pose = viz::makeCameraPose(shiftP, shiftP + pf, pd);

    if (enable)
    {
        window.setViewerPose(cam_pose);
    }
    else
    {

        //    WArrow forward(shiftP,shiftP + pf,0.008,viz::Color::red());window.showWidget(CamName+"Forward",forward);
        //    WArrow downward(shiftP,shiftP + pd,0.008,viz::Color::blue());window.showWidget(CamName+"Downward",downward);

        viz::WCameraPosition camFrame(Vec2f(1, 1), 0.5);
        window.showWidget(CamName + "Frame", camFrame, cam_pose);
        viz::WCameraPosition camAxis(0.5);
        window.showWidget(CamName + "Axis", camAxis, cam_pose);
    }
}

Affine3f getAffine3fPos(Point3d shiftP, Vec3f rotateP)
{
    double x = rotateP[0];
    double y = rotateP[1];
    double z = rotateP[2];

    Point3d pf(cos(y) * cos(z), cos(y) * sin(z), -sin(y));
    y += CV_PI / 2.0;
    Point3d pd(cos(y) * cos(z), cos(y) * sin(z), -sin(y));

    Affine3f cam_pose = viz::makeCameraPose(shiftP, shiftP + pf, pd);

    return cam_pose;
}

void test1(const Mat &cameraMatrix, const Mat &distanceCoeffiecients)
{
    Mat frame;

    vector<int> markerIds;
    vector<vector<Point2f>> markerCorners, rejectedCandicates;
    aruco::DetectorParameters parameters;

    Ptr<aruco::Dictionary> markerDictionary = aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME::DICT_4X4_50);

    VideoCapture vid(0);

    if (!vid.isOpened())
    {
        return;
    }

    vector<Vec3d> rvecs, tvecs;

    while (true)
    {
        if (!vid.read(frame))
        {
            break;
        }

        aruco::detectMarkers(frame, markerDictionary, markerCorners, markerIds);
        aruco::estimatePoseSingleMarkers(markerCorners, arucoSquareDimension, cameraMatrix, distanceCoeffiecients, rvecs, tvecs);

        for (int i = 0; i < markerIds.size(); ++i)
        {

            aruco::drawAxis(frame, cameraMatrix, distanceCoeffiecients, rvecs[i], tvecs[i], 0.025f);
            Mat R = Mat::zeros(3, 3, CV_64F);

            Rodrigues(rvecs[i], R);

            double sy = sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) + R.at<double>(1, 0) * R.at<double>(1, 0));
            double x, y, z;
            if (!(sy < 1e-6))
            {
                x = atan2(R.at<double>(2, 1), R.at<double>(2, 2));
                y = atan2(-R.at<double>(2, 0), sy);
                z = atan2(R.at<double>(1, 0), R.at<double>(0, 0));
            }
            else
            {
                x = atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
                y = atan2(-R.at<double>(2, 0), sy);
                z = 0;
            }

            double rx = x * 180.0 / CV_PI;
            double ry = y * 180.0 / CV_PI;
            double rz = z * 180.0 / CV_PI;

            double distance = ((tvecs[i][2] + 0.02) * 0.00846) * 100;

            cout << ry << "|" << distance << endl;
        }

        cv::imshow("webcam", frame);

        char character = cv::waitKey(1);

        if (character == 27)
        {
            break;
        }
    }
}

Viz3d Win("Win");

class ArucoMarker
{
public:
    int Id = 0;
    double X = 0;
    double Y = 0;
    double DefaultAng = 0;

    bool Detected = false;
    double Ang = 0;
    double RayLen = 0;

    double DisX = 0;
    double DisY = 0;

    ArucoMarker(int id, int x, int y, double ang)
    {
        Id = id;
        X = x;
        Y = y;
        DefaultAng = ang;
    }

    void SetDetectedFlag(bool flag)
    {
        this->Detected = flag;

        if (this->Detected)
        {
            this->DisX = this->DisX + this->RayLen * cos(this->DefaultAng + this->Ang);
            this->DisY = this->DisY + this->RayLen * sin(this->DefaultAng + this->Ang);

            WCylinder cylinder(Point3d(this->X, this->Y, 0), Point3d(this->X, this->Y, 100), 10, 12, Color::red());
            Win.showWidget(string("cylinder") + (to_string(this->Id)), cylinder);
        }
        else
        {
            this->DisX = this->X;
            this->DisY = this->Y;

            WCylinder cylinder(Point3d(this->X, this->Y, 0), Point3d(this->X, this->Y, 100), 10, 12, Color::blue());
            Win.showWidget(string("cylinder") + (to_string(this->Id)), cylinder);
        }

        WLine ray(Point3d(this->X, this->Y, 0), Point3d(this->DisX, this->DisY, 0), Color::green());
        //        WLine ray(Point3d(this->X,this->Y,0),Point3d(this->X,this->Y,0) + 500*Point3d(cos(this->DefaultAng+ this->Ang),sin(this->DefaultAng + this->Ang),0),Color::green());
        ray.setRenderingProperty(LINE_WIDTH, 2);
        Win.showWidget("ray" + to_string(this->Id), ray);
    }

    void SetAngle(double ang)
    {
        this->Ang = ang;
    }
};

// Checks if a matrix is a valid rotation matrix.
bool isRotationMatrix(Mat &R)
{
    Mat Rt;
    transpose(R, Rt);
    Mat shouldBeIdentity = Rt * R;
    Mat I = Mat::eye(3, 3, shouldBeIdentity.type());

    return norm(I, shouldBeIdentity) < 1e-6;
}

// Calculates rotation matrix to euler angles
// The result is the same as MATLAB except the order
// of the euler angles ( x and z are swapped ).
Vec3f rotationMatrixToEulerAngles(Mat &R)
{

    assert(isRotationMatrix(R));

    float sy = sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) + R.at<double>(1, 0) * R.at<double>(1, 0));

    bool singular = sy < 1e-6; // If

    float x, y, z;
    if (!singular)
    {
        x = atan2(R.at<double>(2, 1), R.at<double>(2, 2));
        y = atan2(-R.at<double>(2, 0), sy);
        z = atan2(R.at<double>(1, 0), R.at<double>(0, 0));
    }
    else
    {
        x = atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
        y = atan2(-R.at<double>(2, 0), sy);
        z = 0;
    }
    return Vec3f(x, y, z);
}

void CodeRotateByZ(double x, double y, double thetaz, double &outx, double &outy)
{
    double x1 = x;
    double y1 = y;
    double rz = thetaz * CV_PI / 180;
    outx = cos(rz) * x1 - sin(rz) * y1;
    outy = sin(rz) * x1 + cos(rz) * y1;
}

void CodeRotateByY(double x, double z, double thetay, double &outx, double &outz)
{
    double x1 = x;
    double z1 = z;
    double ry = thetay * CV_PI / 180;
    outx = cos(ry) * x1 + sin(ry) * z1;
    outz = cos(ry) * z1 - sin(ry) * x1;
}
void CodeRotateByX(double y, double z, double thetax, double &outy, double &outz)
{
    double y1 = y;
    double z1 = z;
    double rx = thetax * CV_PI / 180;
    outy = cos(rx) * y1 - sin(rx) * z1;
    outz = cos(rx) * z1 + sin(rx) * y1;
}

int main()
{

    Win.showWidget("Coordinate Widget", viz::WCoordinateSystem());
    Win.setBackgroundColor(Color::white());
    WGrid grid(Vec2i::all(10), Vec2d::all(100), Color::black());

    Win.showWidget("grid", grid);
    grid.setPose(getAffine3fPos(Point3d(500, 500, 0), Vec3f(0, CV_PI / 180.0 * 90.0, 0)));

    AddCamera(Win, "gcam", Point3d(500, -800, 800), Vec3f(0, CV_PI / 180.0 * 35.0, CV_PI / 180.0 * 90.0), true);

    WLine xAxis(Point3d(0, 0, 0), Point3d(1000, 0, 0), Color::red());
    xAxis.setRenderingProperty(LINE_WIDTH, 2);
    Win.showWidget("xAxis", xAxis);

    WLine yAxis(Point3d(0, 0, 0), Point3d(0, 1000, 0), Color::green());
    yAxis.setRenderingProperty(LINE_WIDTH, 2);
    Win.showWidget("yAxis", yAxis);
    //---
    vector<ArucoMarker *> ArucoMarkerVector;

    ArucoMarkerVector.push_back(new ArucoMarker(0, 200, 0, 90 / 180.0 * CV_PI));
    ArucoMarkerVector.push_back(new ArucoMarker(1, 500, 0, 90 / 180.0 * CV_PI));
    ArucoMarkerVector.push_back(new ArucoMarker(2, 800, 0, 90 / 180.0 * CV_PI));

    ArucoMarkerVector.push_back(new ArucoMarker(3, 1000, 200, 180 / 180.0 * CV_PI));
    ArucoMarkerVector.push_back(new ArucoMarker(4, 1000, 500, 180 / 180.0 * CV_PI));
    ArucoMarkerVector.push_back(new ArucoMarker(5, 1000, 800, 180 / 180.0 * CV_PI));

    ArucoMarkerVector.push_back(new ArucoMarker(6, 800, 1000, 270 / 180.0 * CV_PI));
    ArucoMarkerVector.push_back(new ArucoMarker(7, 500, 1000, 270 / 180.0 * CV_PI));
    ArucoMarkerVector.push_back(new ArucoMarker(8, 200, 1000, 270 / 180.0 * CV_PI));

    ArucoMarkerVector.push_back(new ArucoMarker(9, 0, 800, 0 / 180.0 * CV_PI));
    ArucoMarkerVector.push_back(new ArucoMarker(10, 0, 500, 0 / 180.0 * CV_PI));
    ArucoMarkerVector.push_back(new ArucoMarker(11, 0, 200, 0 / 180.0 * CV_PI));

    //---
    for (int i = 0; i < ArucoMarkerVector.size(); ++i)
    {
        ArucoMarker *tmp = ArucoMarkerVector[i];
        WCylinder cylinder(Point3d(tmp->X, tmp->Y, 0), Point3d(tmp->X, tmp->Y, 100), 10, 12, Color::blue());
        Win.showWidget(string("cylinder") + (to_string(tmp->Id)), cylinder);

        WText3D text("No." + to_string(tmp->Id), Point3d(tmp->X, tmp->Y, 0), 20, true, Color::blue());
        Win.showWidget(string("text") + (to_string(tmp->Id)), text);

        WLine angAxis(Point3d(tmp->X, tmp->Y, 0), Point3d(tmp->X, tmp->Y, 0) + 100 * Point3d(cos(tmp->DefaultAng), sin(tmp->DefaultAng), 0), Color::red());
        angAxis.setRenderingProperty(LINE_WIDTH, 5);
        Win.showWidget(string("angAxis") + (to_string(tmp->Id)), angAxis);
    }
    //---

    Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
    Mat distCoefficients;
    loadCameraCalibration("calib", cameraMatrix, distCoefficients);
    Mat frame;
    vector<int> markerIds;
    vector<vector<Point2f>> markerCorners, rejectedCandicates;
    //    aruco::DetectorParameters parameters;
    Ptr<aruco::Dictionary> markerDictionary = aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME::DICT_4X4_50);
    VideoCapture vid(0);
    vector<Vec3d> rvecs, tvecs;

    if (!vid.isOpened())
    {
        return 0;
    }

    while (true)
    {
        if (!vid.read(frame))
        {
            break;
        }

        aruco::detectMarkers(frame, markerDictionary, markerCorners, markerIds);
        aruco::estimatePoseSingleMarkers(markerCorners, arucoSquareDimension, cameraMatrix, distCoefficients, rvecs, tvecs);

        for (int i = 0; i < ArucoMarkerVector.size(); ++i)
        {
            ArucoMarker *tmp = ArucoMarkerVector[i];
            tmp->SetDetectedFlag(false);
        }

        for (int i = 0; i < markerIds.size(); ++i)
        {
            aruco::drawAxis(frame, cameraMatrix, distCoefficients, rvecs[i], tvecs[i], 0.025f);
            //            cout<< markerIds[i] << endl;

            double rm[9];
            Mat RoteM = cv::Mat(3, 3, CV_64FC1, rm);
            Rodrigues(rvecs[i], RoteM);
            double r11 = RoteM.ptr<double>(0)[0];
            double r12 = RoteM.ptr<double>(0)[1];
            double r13 = RoteM.ptr<double>(0)[2];
            double r21 = RoteM.ptr<double>(1)[0];
            double r22 = RoteM.ptr<double>(1)[1];
            double r23 = RoteM.ptr<double>(1)[2];
            double r31 = RoteM.ptr<double>(2)[0];
            double r32 = RoteM.ptr<double>(2)[1];
            double r33 = RoteM.ptr<double>(2)[2];

            double thetaz = atan2(r21, r11) / CV_PI * 180;
            double thetay = atan2(-1 * r31, sqrt(r32 * r32 + r33 * r33)) / CV_PI * 180;
            double thetax = atan2(r32, r33) / CV_PI * 180;

            double tx = tvecs[i][0];
            double ty = tvecs[i][1];
            double tz = tvecs[i][2];

            double x = tx;
            double y = ty;
            double z = tz;

            CodeRotateByZ(x, y, -1 * thetaz, x, y);
            CodeRotateByY(x, z, -1 * thetay, x, z);
            CodeRotateByX(y, z, -1 * thetax, y, z);

            x = x * 1;
            y = y * -1;
            z = z * -1;

            double tmpp = y;
            y = z;
            z = tmpp;

            x *= 1000;
            y *= 1000;
            z *= 1000;

            //            cout<<x<<"|"<<y<<"|"<<z<<endl;

            //            Mat R = Mat::zeros(3,3,CV_64F);
            //            Rodrigues(rvecs[i],R);
            //            Vec3f EulerAngles = rotationMatrixToEulerAngles(R);

            //            double x = EulerAngles[0];
            //            double y = EulerAngles[1];
            //            double z = EulerAngles[2];

            //            double rx = x * 180.0 / CV_PI;
            //            double ry = y * 180.0 / CV_PI;
            //            double rz = z * 180.0 / CV_PI;

            //            double distance = ((tvecs[i][2] + 0.02) * 0.0095413533834586) * 100;

            for (int j = 0; j < ArucoMarkerVector.size(); ++j)
            {
                ArucoMarker *tmp = ArucoMarkerVector[j];

                if (tmp->Id == markerIds[i])
                {
                    //                    tmp->SetAngle(y);
                    //                    tmp->RayLen = distance * 1000;
                    //                    tmp->SetDetectedFlag(true);

                    //                    tmp->X = 0;
                    //                    tmp->Y = 0;
                    double a = tmp->DefaultAng - CV_PI / 2;

                    double disX = tmp->X + x * cos(a) - y * sin(a);
                    double disY = tmp->Y + y * cos(a) + x * sin(a);
                    double disZ = z;

                    WCylinder cylinder(Point3d(tmp->X, tmp->Y, 0), Point3d(tmp->X, tmp->Y, 100), 10, 12, Color::red());
                    Win.showWidget(string("cylinder") + (to_string(tmp->Id)), cylinder);

                    WLine ray(Point3d(tmp->X, tmp->Y, 0), Point3d(disX, disY, disZ), Color::green());
                    ray.setRenderingProperty(LINE_WIDTH, 2);
                    Win.showWidget("ray" + to_string(tmp->Id), ray);

                    WLine rayv(Point3d(disX, disY, 0), Point3d(disX, disY, disZ), Color::blue());
                    rayv.setRenderingProperty(LINE_WIDTH, 2);
                    Win.showWidget("rayv" + to_string(tmp->Id), rayv);

                    WLine rayh(Point3d(tmp->X, tmp->Y, 0), Point3d(disX, disY, 0), Color::red());
                    rayh.setRenderingProperty(LINE_WIDTH, 2);
                    Win.showWidget("rayh" + to_string(tmp->Id), rayh);

                    break;
                }
            }
        }

        cv::imshow("webcam", frame);
        char character = cv::waitKey(1);

        if (character == 27)
        {
            break;
        }

        Win.spinOnce(1, true);
    }
    return 0;
}

int main_v()
{
    Viz3d Win("Win");
    Win.showWidget("Coordinate Widget", viz::WCoordinateSystem());
    Win.setBackgroundColor(Color::white());
    WGrid grid(Vec2i::all(50), Vec2d::all(1), Color::black());
    Win.showWidget("grid", grid);

    Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
    Mat distanceCoefficients;
    loadCameraCalibration("calib", cameraMatrix, distanceCoefficients);
    Mat frame;
    vector<int> markerIds;
    vector<vector<Point2f>> markerCorners, rejectedCandicates;
    aruco::DetectorParameters parameters;
    Ptr<aruco::Dictionary> markerDictionary = aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME::DICT_4X4_50);
    VideoCapture vid(0);

    if (!vid.isOpened())
    {
        return -1;
    }

    vector<Vec3d> rvecs, tvecs;

    while (true)
    {
        if (!vid.read(frame))
        {
            break;
        }

        aruco::detectMarkers(frame, markerDictionary, markerCorners, markerIds);
        aruco::estimatePoseSingleMarkers(markerCorners, arucoSquareDimension, cameraMatrix, distanceCoefficients, rvecs, tvecs);

        for (int i = 0; i < markerIds.size(); ++i)
        {
            aruco::drawAxis(frame, cameraMatrix, distanceCoefficients, rvecs[i], tvecs[i], 0.025f);

            WLine rvl(Point3d(0, 0, 0), Point3d(rvecs[i][0], rvecs[i][1], rvecs[i][2]), Color::green());
            rvl.setRenderingProperty(LINE_WIDTH, 2);
            Win.showWidget("rvl", rvl);
        }

        cv::imshow("webcam", frame);

        char character = cv::waitKey(1);

        if (character == 27)
        {
            break;
        }

        Win.spinOnce(1, true);
    }

    return 0;
}

int main_p()
{
    Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
    Mat distanceCoefficients;
    loadCameraCalibration("calib", cameraMatrix, distanceCoefficients);

    startWebcamMonitoring(cameraMatrix, distanceCoefficients);
    return 0;
}

int main_c()
{
    Mat frame;
    Mat drawToFrame;

    Mat cameraMatrix = Mat::eye(3, 3, CV_64F);

    Mat distanceCoefficients;

    vector<Mat> savedImages;

    vector<vector<Point2f>> makerCorners, rejectedCandidates;

    VideoCapture vid(0);

    if (!vid.isOpened())
    {
        return 0;
    }

    while (true)
    {
        if (!vid.read(frame))
        {
            break;
        }

        vector<Vec2f> foundPoints;

        bool found = false;

        found = findChessboardCorners(frame, chessboardDimension, foundPoints, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);

        frame.copyTo(drawToFrame);

        drawChessboardCorners(drawToFrame, chessboardDimension, foundPoints, found);

        if (found)
        {
            cv::imshow("webcam", drawToFrame);
        }
        else
        {
            cv::imshow("webcam", frame);
        }

        char character = cv::waitKey(1);

        switch (character)
        {
        case ' ': //save img
            if (found)
            {
                Mat temp;
                frame.copyTo(temp);
                savedImages.push_back(temp);
                printf("%d\n", savedImages.size());
            }
            break;
        case 13: //start calib
            if (savedImages.size() > 15)
            {
                cameraCalibration(savedImages, chessboardDimension, calibrationSquareDimension, cameraMatrix, distanceCoefficients);
                saveCameraCalibaration("calib", cameraMatrix, distanceCoefficients);
            }
            break;
        case 27: //exit
            return 0;
            break;
        }

        if (character == 'c')
        {
            break;
        }
    }

    vid.release();

    return 0;
}

int main_rc()
{
    Mat frame;
    Mat drawToFrame;

    Mat cameraMatrix = Mat::eye(3, 3, CV_64F);

    Mat distanceCoefficients;

    vector<Mat> savedImages;

    vector<vector<Point2f>> makerCorners, rejectedCandidates;

    for (int i = 0; i < 177; ++i)
    {
        string str = "/home/pi/QtProjects/Aruco/cimgs/";
        str.append(to_string(i));
        str.append(".png");

        Mat img = imread(str, IMREAD_COLOR);
        //        cv::imshow("t",img);
        //        cv::waitKey(100);

        Mat temp;
        img.copyTo(temp);
        savedImages.push_back(temp);
    }

    cameraCalibration(savedImages, chessboardDimension, calibrationSquareDimension, cameraMatrix, distanceCoefficients);
    saveCameraCalibaration("calib", cameraMatrix, distanceCoefficients);

    return 0;
}
