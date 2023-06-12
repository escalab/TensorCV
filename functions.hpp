#pragma once

#include <iostream>
#include <vector>
#include <cublas_v2.h>
#include <mma.h>
#include <tuple>

#include "common.hpp"

using namespace cv;
namespace tensorcv {

void test();

typedef struct splitted_src_ {
    half* R = NULL;
    half* G = NULL;
    half* B = NULL;
} splitted_src;

half* upload ( Mat* src, int rows, int cols );
half* upload ( int rows, int cols );
Mat download ( half* output_d, int rows, int cols, int transpose=0 );
void upload_split ( splitted_src*, Mat* src, int rows, int cols );
void upload_split ( splitted_src*, int rows, int cols );
Mat download_merge ( half* output1_d, half* output2_d, half* output3_d, int rows, int cols, int transpose=0 );
void synch();

void free (half*, half*);
void free (splitted_src, splitted_src);

class imgprocKernel {
    public: 
        imgprocKernel(){}
        ~imgprocKernel(){}

        void init_resize(int iRow_, int iCol_, int oRow_, int oCol_);
        void upload_resize();
        void apply_resize(cublasHandle_t handle, half* src, half* dst);
        void release_resize();

        void init_crop(int iRow_, int iCol_, int oRow_, int oCol_);
        void upload_crop();
        void apply_crop(cublasHandle_t handle, half* src, half* dst);
        void release_crop();

        void init_cvtcolor(int iRow_, int iCol_, int colorCode);
        void upload_cvtcolor();
        void apply_cvtcolor(cublasHandle_t handle, half* src, half* dst);
        void release_cvtcolor();

        void init_rotate(int iRow_, int iCol_, int repeat);
        void upload_rotate();
        void apply_rotate(cublasHandle_t handle, half*, half*, half*, half*, half*, half*);
        void release_rotate();

        void init_normalize(int iRow_, int iCol_, int channelCode);
        void upload_normalize();
        // void comptue_norm(half* sum, half* ssum, half* src);
        void apply_normalize(cublasHandle_t handle, half*, half*, half*, half*, half*, half*);
        void release_normalize();

        void init_integrated(int iRow, int iCol, int rRow, int rCol, int cRow, int cCol, int colorCode, int rotate);
        void upload_integrated();
        void apply_integrated(cublasHandle_t handle, half* src, half* dst);
        void release_integrated();

    private:
        unsigned iRow, iCol;
        unsigned oRow, oCol;

        float rowFactor, colFactor;
        unsigned cropRow=0, cropCol=0;

        int repeat=0;
        int channelCode=0;
        
        half* kernel1 = NULL;
        half* kernel2 = NULL;
        half* kernel3 = NULL;
        half* kernel4 = NULL;
        half* kernel5 = NULL;
        half* d_kernel1 = NULL;
        half* d_kernel2 = NULL;
        half* d_kernel3 = NULL;
        half* d_kernel4 = NULL;
        half* d_kernel5 = NULL;
        
        half* d_temp1 = NULL;
        half* d_temp2 = NULL;
        half* d_temp3 = NULL;
        // half* d_temp4 = NULL;
        // half* d_temp5 = NULL;
        // half* d_temp6 = NULL;
        // half* d_temp7 = NULL;
        // half* d_temp8 = NULL;
        // half* d_temp9 = NULL;
};

}