
#ifndef FACE_ALIGNMENT_H
#define FACE_ALIGNMENT_H

#include <string>

namespace fa{

class FaceAlignment
{
public:
    FaceAlignment(const std::string&,const std::string&);
    ~FaceAlignment();

    int Run();

private:
    int ParseConfigFile();
    int AlignAndCropFace();

private:
    std::string strdstimg;
    std::string strconf;

    std::string strname;
    std::string strworkpath;
    std::string strdataset;
    std::string strsavepath;
    std::string strfaceCascadeFilename;
    std::string strflandmarkModel;

    bool m_imgabusolutepath;
};

#endif

}