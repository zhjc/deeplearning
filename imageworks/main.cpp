
#include "core/include/opencv2/core/core.hpp"
#include "highgui/include/opencv2/highgui/highgui.hpp"
#include "imgproc/include/opencv2/imgproc/imgproc.hpp"
#include "video/include/opencv2/video/tracking.hpp"
#include "objdetect/include/opencv2/objdetect/objdetect.hpp"

#include "libflandmark/flandmark_detector.h"

#include "include/opencv/cv.h"

//#include <windows.h>

#include <fstream>
#include <iostream>

using namespace cv;
using namespace std;

int main_tt_cam()
{
    //声明IplImage指针  
    IplImage* pFrame = NULL;

    //获取摄像头  
    CvCapture* pCapture = cvCreateCameraCapture(-1);

    //创建窗口  
    cvNamedWindow("video", 1);

    //显示视屏  
    while (1)
    {
        pFrame = cvQueryFrame(pCapture);
        if (!pFrame)break;
        cvShowImage("video", pFrame);
        char c = cvWaitKey(33);
        if (c == 27)break;
    }
    cvReleaseCapture(&pCapture);
    cvDestroyWindow("video");

    return 0;
}

void detectAndDisplay(Mat& frame, CascadeClassifier& face_cascade){
    std::vector<Rect> faces;
    Mat frame_gray;

    cvtColor(frame, frame_gray, CV_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);

    face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

    for (int i = 0; i < faces.size(); i++){
        Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
        ellipse(frame, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
    }

    imshow("img", frame);
    waitKey(0);
}

int main_hh()
{
    VideoCapture cap(0); 
    if (!cap.isOpened())
    {
        return -1;
    }

    Mat edges;
    string face_cascade_name = "D:\\Master\\Public\\deeplearning\\deeplearning\\data\\haarcascade_frontalface_alt2.xml";
    CascadeClassifier face_cascade;

    if (!face_cascade.load(face_cascade_name)){
        printf("[error] 无法加载级联分类器文件！\n");
        return -1;
    }

    int nTick = 0;
    for (;;)
    {
        if (!cap.isOpened())
        {//等等摄像头打开  
            continue;
        }

        Mat frame;
        nTick = getTickCount();
        cap >> frame; // get a new frame from camera  
        if (frame.data == NULL)
        {//等到捕获到数据  
            continue;
        }
        cvtColor(frame, edges, CV_BGR2BGRA);

        detectAndDisplay(edges, face_cascade);

        if (waitKey(30) >= 0) break;
    }
        
    return 0;
}

int detectFaceInImage(IplImage* orig, IplImage* input, CvHaarClassifierCascade* cascade, \
    FLANDMARK_Model *model, int *bbox, double *landmarks);
void FaceAlign(const Mat &orig, double *landmarks, Mat& outputarray);
void cropAlignedFace(Mat *input, CvHaarClassifierCascade* cascade, string saveAddr);

int main(int argc, char** argv)
{
    //wchar_t buf[1000];
    //GetCurrentDirectory(1000, buf);
    
    //wstring ws = buf;
    //cout << ws.c_str() << endl;

    //main_hh();
    //main_tt_cam();

    string strWorkPath = "D:\\Master\\Public\\deeplearning\\deeplearning\\";

    // Haar Cascade file, used for Face Detection.
    const char faceCascadeFilename[] = "D:\\Master\\Public\\deeplearning\\deeplearning\\data\\haarcascade_frontalface_alt.xml";
    // ***face detection ***//
    CvHaarClassifierCascade* faceCascade;
    faceCascade = (CvHaarClassifierCascade*)cvLoad(faceCascadeFilename);
    if (!faceCascade)
    {
        printf("Couldn't load Face detector '%s'\n", faceCascadeFilename);
        exit(1);
    }
    // begin flandmark load model
    FLANDMARK_Model * model = flandmark_init("D:\\Master\\Public\\deeplearning\\deeplearning\\data\\flandmark_model.dat");
    if (model == 0)
    {
        printf("Structure model wasn't created. Corrupted file flandmark_model.dat?\n");
        exit(1);
    }

    // *** input image *** //
    int *bbox_src = (int*)malloc(4 * sizeof(int));  //memory allocation---bbox
    double *landmarks_src = (double*)malloc(2 * model->data.options.M*sizeof(double)); //landmarks
    Mat output;
    // 
    string txtName = strWorkPath + "data\\undetected_face.txt";        //save the undetected face to the txt file
    std::ofstream fout;
    fout.open(txtName.c_str());
    string saveFolder = strWorkPath + "data\\face_aligned\\";  //the directory to save the aligned images

    vector<string> flist;
    //flist.push_back("DSCF4600_zoufh.jpg");
    //flist.push_back("test.JPG");
    flist.push_back("test4.png");
    //GetAllFilesInCurFolder(foldname.c_str(), suffix, flist);
    //*** operates on the image ***//
    for (int i = 0; i < flist.size(); i++)
    {
        string imgAddr = strWorkPath + "data\\face_captured\\" + flist[i];
        Mat src_mat = imread(imgAddr);
        string saveFile = saveFolder + flist[i];
        IplImage *src = cvLoadImage(imgAddr.c_str());
        IplImage *src_gray = cvCreateImage(cvSize(src->width, src->height), IPL_DEPTH_8U, 1);
        if (src == NULL)
        {
            fprintf(stderr, "Cannot open image %s. Exiting...\n", imgAddr);
            continue;
        }
        cvConvertImage(src, src_gray);  // convert image to grayscale
        // *** detect landmarks and bbox ***//
        int ret_src = detectFaceInImage(src, src_gray, faceCascade, model, bbox_src, landmarks_src);
        if (ret_src != 0) {
            printf("Landmark not detected!\n");
            fout << flist[i] << "\n";
            continue;
        }
        // *** face alignment begin *** //
        FaceAlign(src_mat, landmarks_src, output);

        // *** crop the aligned face *** //
        cropAlignedFace(&output, faceCascade, saveFile);
    }
    fout.close();

    // *** clean up *** //
    free(bbox_src);
    free(landmarks_src);
    cvDestroyAllWindows();
    cvReleaseHaarClassifierCascade(&faceCascade);
    flandmark_free(model);
}

//detect the face bbox and landmarks in the image
int detectFaceInImage(IplImage* orig, IplImage* input, CvHaarClassifierCascade* cascade, \
    FLANDMARK_Model *model, int *bbox, double *landmarks)
{
    int ret = 0;
    // Smallest face size.
    CvSize minFeatureSize = cvSize(50, 50);
    int flags = CV_HAAR_DO_CANNY_PRUNING;
    // How detailed should the search be.
    float search_scale_factor = 1.1f;
    CvMemStorage* storage;
    CvSeq* rects;
    int nFaces;

    storage = cvCreateMemStorage(0);
    cvClearMemStorage(storage);

    // Detect all the faces in the greyscale image.
    rects = cvHaarDetectObjects(input, cascade, storage, search_scale_factor, 2, flags, minFeatureSize);
    nFaces = rects->total; //the ammounts of face in the image
    if (nFaces <= 0)
    {
        printf("NO Face\n");
        ret = -1;
        return ret;
    }

    CvRect *r = (CvRect*)cvGetSeqElem(rects, 0);
    printf("Detected %d faces\n", nFaces);
    //If there is more than 1 face in picture, we select the biggest face 
    if (nFaces > 1)
    {
        for (int iface = 1; iface < nFaces; ++iface)
        {
            CvRect *rr = (CvRect*)cvGetSeqElem(rects, iface);
            if (rr->width > r->width)
                *r = *rr;
        }
    }

    bbox[0] = r->x;
    bbox[1] = r->y;
    bbox[2] = r->x + r->width;
    bbox[3] = r->y + r->height;

    ret = flandmark_detect(input, bbox, model, landmarks);

    //Display landmarks
    //cvRectangle(orig, cvPoint(bbox[0], bbox[1]), cvPoint(bbox[2], bbox[3]), CV_RGB(255, 0, 0));
    //for (int i = 0; i < 2 * model->data.options.M; i += 2)
    //{
    //  cvCircle(orig, cvPoint(int(landmarks[i]), int(landmarks[i + 1])), 3, CV_RGB(255, 0, 0));
    //}
    cvReleaseMemStorage(&storage);
    return ret;
}

//Face Alignment
void FaceAlign(const Mat &orig, double *landmarks, Mat& outputarray)
{
    int desiredFaceWidth = orig.cols;
    int desiredFaceHeight = desiredFaceWidth;

    // Get the eyes center-point with the landmarks
    Point2d leftEye = Point2d((landmarks[2] + landmarks[10]) * 0.5f, (landmarks[3] + landmarks[11]) * 0.5f);
    Point2d rightEye = Point2d((landmarks[4] + landmarks[12]) * 0.5f, (landmarks[5] + landmarks[13]) * 0.5f);;

    // Get the center between the 2 eyes center-points
    Point2f eyesCenter = Point2f((leftEye.x + rightEye.x) * 0.5f, (leftEye.y + rightEye.y) * 0.5f);

    // Get the angle between the line eyes and horizontal line.
    double dy = (rightEye.y - leftEye.y);
    double dx = (rightEye.x - leftEye.x);
    double len = sqrt(dx*dx + dy*dy);
    double angle = atan2(dy, dx) * 180.0 / CV_PI; // Convert from radians to degrees.
    double scale = 1;
    // Get the transformation matrix for rotating and scaling the face to the desired angle & size.
    Mat rot_mat = getRotationMatrix2D(eyesCenter, angle, scale);
    outputarray.create(desiredFaceHeight, desiredFaceWidth, CV_8UC3);
    warpAffine(orig, outputarray, rot_mat, outputarray.size());
    return;
}

//Crop the aligned face
void cropAlignedFace(Mat *input, CvHaarClassifierCascade* cascade, string saveAddr)
{
    CvSize minFeatureSize = cvSize(50, 50);
    int flags = CV_HAAR_DO_CANNY_PRUNING;
    float search_scale_factor = 1.1f;
    CvMemStorage* storage;
    CvSeq* rects;
    storage = cvCreateMemStorage(0);
    cvClearMemStorage(storage);

    // Detect all the faces in the already aligned image.
    IplImage ipl_img(*input);
    IplImage *img = &ipl_img;
    rects = cvHaarDetectObjects(img, cascade, storage, search_scale_factor, 2, flags, minFeatureSize);
    int nfaces = rects->total;
    if (nfaces <= 0)
        return;
    CvRect *r = (CvRect*)cvGetSeqElem(rects, 0);
    //choose the biggest face
    if (nfaces > 1)
    {
        for (int iface = 1; iface < nfaces; ++iface)
        {
            CvRect *rr = (CvRect*)cvGetSeqElem(rects, iface);
            if (rr->width > r->width)
                *r = *rr;
        }
    }
    // Get the new bounding box with the scale  
    float scale = 0.1;            //the scale to expand the bbox size
    int paddingx = scale * r->width;
    int paddingy = scale * r->height;
    int newx = max(0, r->x - paddingx);
    int newy = max(0, r->y - paddingy);
    int newwidth = min(input->cols - 1, r->width + paddingx * 2);
    int newheight = min(input->rows - 1, r->height + paddingy * 2);
    //crop, and save the cropped face 
    cvSetImageROI(img, cvRect(newx, newy, newwidth, newheight));
    cvSaveImage(saveAddr.c_str(), img);
    return;
}

int test_cropface()
{
    // Read the images to be aligned
    Mat im1 = imread("D:/Master/Public/deeplearning/deeplearning/data/face_captured/DSCF4596_zoufh.jpg");
    Mat im2 = imread("D:/Master/Public/deeplearning/deeplearning/data/face_captured/DSCF4600_zoufh.jpg");

    /*
    imshow("Image 1", im1);
    waitKey(0);

    imshow("Image 1", im2);
    waitKey(0);*/

    // Convert images to gray scale;
    Mat im1_gray, im2_gray;
    cvtColor(im1, im1_gray, CV_BGR2GRAY);
    cvtColor(im2, im2_gray, CV_BGR2GRAY);

    imshow("Image 1", im1_gray);
    waitKey(0);
    // Define the motion model
    const int warp_mode = MOTION_AFFINE;
    // Set a 2x3 or 3x3 warp matrix depending on the motion model.
    Mat warp_matrix;
    // Initialize the matrix to identity
    if (warp_mode == MOTION_HOMOGRAPHY)
        warp_matrix = Mat::eye(3, 3, CV_32F);
    else
        warp_matrix = Mat::eye(2, 3, CV_32F);
    // Specify the number of iterations.
    int number_of_iterations = 5000;
    // Specify the threshold of the increment
    // in the correlation coefficient between two iterations
    double termination_eps = 1e-4;
    // Define termination criteria
    TermCriteria criteria(TermCriteria::COUNT + TermCriteria::EPS, number_of_iterations, termination_eps);
    // Run the ECC algorithm. The results are stored in warp_matrix.

    findTransformECC(
        im1_gray,
        im2_gray,
        warp_matrix,
        warp_mode,
        criteria
        );
    // Storage for warped image.
    Mat im2_aligned;
    if (warp_mode != MOTION_HOMOGRAPHY)
        // Use warpAffine for Translation, Euclidean and Affine
        warpAffine(im2, im2_aligned, warp_matrix, im1.size(), INTER_LINEAR + WARP_INVERSE_MAP);
    else
        // Use warpPerspective for Homography
        warpPerspective(im2, im2_aligned, warp_matrix, im1.size(), INTER_LINEAR + WARP_INVERSE_MAP);
    // Show final result
    //imshow("Image 1", im1);
    //imshow("Image 2", im2);
    imshow("Image 2 Aligned", im2_aligned);
    waitKey(0);

    return 0;
}