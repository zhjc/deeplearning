
#include "face_alignment.h"
#include <iostream>

using namespace std;

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        cout << "please input right arguments!" << endl;
        return -1;
    }

    string strConfigPath = argv[1];
    string dstImageFile;
    if (argc == 3)
    {
        dstImageFile = argv[2];
    }
    else if (argc < 3)
    {
        cout << "please input enough arguments!" << endl;
    }

    fa::FaceAlignment fa(strConfigPath, dstImageFile);
    int rs = fa.Run();
    if (rs < 0)
    {
        cout << "detect or crop failed!" << endl;
    }

    return 0;
}