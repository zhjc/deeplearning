
#include "face_alignment.h"

#include "core/include/opencv2/core/core.hpp"
#include "highgui/include/opencv2/highgui/highgui.hpp"
#include "imgproc/include/opencv2/imgproc/imgproc.hpp"
#include "video/include/opencv2/video/tracking.hpp"
#include "objdetect/include/opencv2/objdetect/objdetect.hpp"

#include "libflandmark/flandmark_detector.h"

#include "include/opencv/cv.h"

#include "json/json.h"

#include <fstream>
#include <iostream>

using namespace cv;
using namespace std;

namespace fa{

    int detectFaceInImage(IplImage* orig, IplImage* input, CvHaarClassifierCascade* cascade, \
        FLANDMARK_Model *model, int *bbox, double *landmarks);
    void FaceAlign(const Mat &orig, double *landmarks, Mat& outputarray);
    void cropAlignedFace(Mat *input, CvHaarClassifierCascade* cascade, string saveAddr);

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

    FaceAlignment::FaceAlignment(const std::string& strconfigPath, const string& dstImageFile)
        : strconf(strconfigPath)
        , strdstimg(dstImageFile)
        , m_imgabusolutepath(false)
    {
    }

    FaceAlignment::~FaceAlignment()
    {
    }

    int FaceAlignment::AlignAndCropFace()
    {
        // Haar Cascade file, used for Face Detection.
        const string strCasFile = strworkpath + strfaceCascadeFilename;
        const char* faceCascadeFilename = strCasFile.c_str();
        // ***face detection ***//
        CvHaarClassifierCascade* faceCascade;
        faceCascade = (CvHaarClassifierCascade*)cvLoad(faceCascadeFilename);
        if (!faceCascade)
        {
            printf("Couldn't load Face detector '%s'\n", faceCascadeFilename);
            exit(1);
        }
        // begin flandmark load model
        FLANDMARK_Model * model = flandmark_init((strworkpath + strflandmarkModel).c_str());
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
        string txtName = strworkpath + "data\\undetected_face.txt";        //save the undetected face to the txt file
        std::ofstream fout;
        fout.open(txtName.c_str());
        string saveFolder = strworkpath + strsavepath;  //the directory to save the aligned images

        vector<string> flist;
        if (strdstimg != "")
        {
            flist.push_back(strdstimg);
        }
        //flist.push_back("DSCF4600_zoufh.jpg");
        //flist.push_back("test.JPG");

        /*fstream in(strworkpath + "data\\filename.txt");
        if (!in.is_open())
        {
        printf("open file list file failed!\n");
        exit(1);
        }
        else
        {
        string fileitem;
        while (in >> fileitem)
        {
        flist.push_back(fileitem);
        }
        flist.push_back("DSCF46041_zhangjc.JPG");
        flist.push_back("DSCF46042_zhangjc.JPG");
        flist.push_back("DSCF46043_zhangjc.JPG");
        }*/

        //flist.push_back("test4.png");

        //*** operates on the image ***//
        for (size_t i = 0; i < flist.size(); i++)
        {
            string imgAddr = m_imgabusolutepath ? flist[i] : (strworkpath + strdataset + flist[i]);
            Mat src_mat = imread(imgAddr);
            
            IplImage *src = cvLoadImage(imgAddr.c_str());
            if (src == NULL)
            {
                fprintf(stderr, "Cannot open image %s. Exiting...\n", imgAddr);
                continue;
            }

            IplImage *src_gray = cvCreateImage(cvSize(src->width, src->height), IPL_DEPTH_8U, 1);

            cvConvertImage(src, src_gray);  // convert image to gray scale
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
            string saveFile = saveFolder + flist[i];
            if (m_imgabusolutepath)
            {
                
                saveFile = flist[i];
                string::size_type index = saveFile.find_last_of("\\");
                string extname = saveFile.substr(index + 1, saveFile.size() - 1);
                saveFile = saveFolder+extname;
            }
            cropAlignedFace(&output, faceCascade, saveFile);
        }
        fout.close();

        // *** clean up *** //
        free(bbox_src);
        free(landmarks_src);
        cvDestroyAllWindows();
        cvReleaseHaarClassifierCascade(&faceCascade);
        flandmark_free(model);

        return 0;
    }

    int FaceAlignment::Run()
    {
        int st = 0;

        st = ParseConfigFile();
        if (st != 0)
        {
            return st;
        }

        st = AlignAndCropFace();

        return st;
    }

    int FaceAlignment::ParseConfigFile()
    {
        Json::Reader reader;
        Json::Value root;

        ifstream in(strconf);
        if (in.is_open())
        {
            if (reader.parse(in, root))
            {
                if (!root["name"].isNull())
                {
                    strname = root["name"].asString();
                }

                if (!root["workpath"].isNull())
                {
                    strworkpath = root["workpath"].asString();
                }

                if (!root["imgabusolutepath"].isNull())
                {
                    string strabu = root["imgabusolutepath"].asString();
                    if (strabu == "true")
                    {
                        m_imgabusolutepath = true;
                    }
                }

                if (!root["config"].isNull())
                {
                    int configsize = root["config"].size();
                    for (int i = 0; i < configsize; ++i)
                    {
                        if (!root["config"][i]["dataset"].isNull())
                        {
                            strdataset = root["config"][i]["dataset"].asString();
                        }

                        if (!root["config"][i]["savepath"].isNull())
                        {
                            strsavepath = root["config"][i]["savepath"].asString();
                        }

                        if (!root["config"][i]["faceCascadeFilename"].isNull())
                        {
                            strfaceCascadeFilename = root["config"][i]["faceCascadeFilename"].asString();
                        }

                        if (!root["config"][i]["flandmarkModel"].isNull())
                        {
                            strflandmarkModel = root["config"][i]["flandmarkModel"].asString();
                        }
                    }
                }
            }
            else
            {
                return -2;
            }
        }
        else
        {
            return -2;
        }

        return 0;
    }

}