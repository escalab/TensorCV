#include "cvTest.hpp"

int main(int argc, char** argv )
{
    if ( argc != 1 ) {
        printf("usage: ./cvTest\n");
        return -1;
    }

    // check GPUs
    std::cout << "We have " << cuda::getCudaEnabledDeviceCount() << " GPUs" << "\n";

    int long long execution_time = 0;
    int numImg = 100;

    // mode 0 : CPU only 
    // mode 1 : GPU only
    // mode 2 : TensorCV only
    // mode 3 : all
    int mode = 3;

// ***************************************************************************************************

    std::cout << "OpenCV img preprocessing with CPU" << std::endl;
    std::cout << "Iter,load,resiz,center,cvt,rotate,norm" << std::endl;

    Mat outputImg, smallImg, outputImg_rgb[3];

    if (mode == 0 || mode == 3) {
        for (int step=0; step<numImg; step++) {
            std::cout << step << ",";

            // load image from ../img/
            auto blockStart = high_resolution_clock::now();
            char filepath[32];
            sprintf(filepath, "../img/input%d.jpg", step%20+1);
            Mat inputImg = imread(filepath, IMREAD_COLOR );
            if ( !inputImg.data ){ printf("No image data \n"); return -1;}
            std::cout << duration_cast<nanoseconds>(high_resolution_clock::now()-blockStart).count() << ",";

            // resize
            blockStart = high_resolution_clock::now();
            Size resizeSize(256,256);
            resize(inputImg, smallImg, resizeSize, 0, 0, INTER_LINEAR);
            std::cout << duration_cast<nanoseconds>(high_resolution_clock::now()-blockStart).count() << ",";

            // center crop
            blockStart = high_resolution_clock::now();
            Rect rect(16,16,224,224);
            outputImg = smallImg(rect);
            std::cout << duration_cast<nanoseconds>(high_resolution_clock::now()-blockStart).count() << ",";

            // cvt color
            blockStart = high_resolution_clock::now();
            // COLOR_BGR2GRAY, COLOR_BGR2RGB, COLOR_RGB2YCrCb, ...
            cvtColor(outputImg, outputImg, COLOR_RGB2YCrCb, 0);
            std::cout << duration_cast<nanoseconds>(high_resolution_clock::now()-blockStart).count() << ",";

            // rotate
            blockStart = high_resolution_clock::now();
            rotate(outputImg, outputImg, ROTATE_90_CLOCKWISE);
            std::cout << duration_cast<nanoseconds>(high_resolution_clock::now()-blockStart).count() << ",";
            
            // normalize
            blockStart = high_resolution_clock::now();
            split(outputImg, outputImg_rgb);
            normalize(outputImg_rgb[0], outputImg_rgb[0], 1.0, 0.0, NORM_L2, CV_32F);
            normalize(outputImg_rgb[1], outputImg_rgb[1], 1.0, 0.0, NORM_L2, CV_32F);
            normalize(outputImg_rgb[2], outputImg_rgb[2], 1.0, 0.0, NORM_L2, CV_32F);
            merge(outputImg_rgb, 3, outputImg);
            std::cout << duration_cast<nanoseconds>(high_resolution_clock::now()-blockStart).count() << "\n";

            inputImg.release();
        }
        // imwrite("output.jpg", outputImg);
    }

// ***************************************************************************************************

    std::cout << "OpenCV img preprocessing with GPU" << std::endl;
    std::cout << "Iter,load,resiz,center,cvt,rotate,norm" << std::endl;

    cuda::GpuMat inputImg_gpu, outputImg_gpu, outputImg_gpu_rgb[3];

    if (mode == 1 || mode == 3) {
        for (int step=0; step<numImg; step++) {
            std::cout << step << ",";

            // load image from ../img/
            auto blockStart = high_resolution_clock::now();
            char filepath[32];
            sprintf(filepath, "../img/input%d.jpg", step%20+1);
            Mat inputImg = imread(filepath, IMREAD_COLOR );
            if ( !inputImg.data ){ printf("No image data \n"); return -1;}
            inputImg_gpu.upload(inputImg);
            std::cout << duration_cast<nanoseconds>(high_resolution_clock::now()-blockStart).count() << ",";

            // resize
            blockStart = high_resolution_clock::now();
            Size resizeSize(256,256);
            cuda::resize(inputImg_gpu, outputImg_gpu, resizeSize, 0, 0, INTER_LINEAR);
            std::cout << duration_cast<nanoseconds>(high_resolution_clock::now()-blockStart).count() << ",";

            // center crop
            blockStart = high_resolution_clock::now();
            Rect rect(16,16,224,224);
            outputImg_gpu = outputImg_gpu(rect);
            std::cout << duration_cast<nanoseconds>(high_resolution_clock::now()-blockStart).count() << ",";
        
            // cvt color
            blockStart = high_resolution_clock::now();
            // COLOR_BGR2GRAY, COLOR_BGR2RGB, COLOR_RGB2YCrCb, ...
            cuda::cvtColor(outputImg_gpu, outputImg_gpu, COLOR_RGB2YCrCb, 0);
            std::cout << duration_cast<nanoseconds>(high_resolution_clock::now()-blockStart).count() << ",";

            // rotate
            blockStart = high_resolution_clock::now();
            Size rotateSize(224,224);
            cuda::rotate(outputImg_gpu, outputImg_gpu, rotateSize, 90, 0, 224);
            std::cout << duration_cast<nanoseconds>(high_resolution_clock::now()-blockStart).count() << ",";

            // normalizaiton
            blockStart = high_resolution_clock::now();
            cuda::split(outputImg_gpu, outputImg_gpu_rgb);
            cuda::normalize(outputImg_gpu_rgb[0], outputImg_gpu_rgb[0], 1.0, 0.0, NORM_L2, CV_32F);
            cuda::normalize(outputImg_gpu_rgb[1], outputImg_gpu_rgb[1], 1.0, 0.0, NORM_L2, CV_32F);
            cuda::normalize(outputImg_gpu_rgb[2], outputImg_gpu_rgb[2], 1.0, 0.0, NORM_L2, CV_32F);
            cuda::merge(outputImg_gpu_rgb, 3, outputImg_gpu);
            std::cout << duration_cast<nanoseconds>(high_resolution_clock::now()-blockStart).count() << "\n";

            outputImg_gpu.download(outputImg);
            inputImg.release();
        }
        // imwrite("output.jpg", outputImg);
    }

    smallImg.release();
    inputImg_gpu.release();
    outputImg_gpu.release();
    outputImg_gpu_rgb[0].release();
    outputImg_gpu_rgb[1].release();
    outputImg_gpu_rgb[2].release();
    
// ***************************************************************************************************

    tensorcv::imgprocKernel kernel;

    cublasHandle_t handle;
    cublasCreate(&handle);

    std::cout << "tensorcv img preprocessing with GPU" << std::endl;
    std::cout << "Iter,load,resiz,center,cvt,rotate,norm" << std::endl;
    long int execTime[numImg][7] = {0};

    if (mode == 2 || mode == 3) {
        kernel.init_resize(4032, 3024, 256, 256);
        kernel.upload_resize();
        for (int step=0; step<numImg; step++) {

            // load image from ../img/
            auto blockStart = high_resolution_clock::now();
            char filepath[32];
            sprintf(filepath, "../img/input%d.jpg", step%20+1);
            Mat inputImg = imread(filepath, IMREAD_COLOR );
            if ( !inputImg.data ){ printf("No image data \n"); return -1;}
            half* d_inputImgArr = tensorcv::upload(&inputImg, inputImg.rows, inputImg.cols);
            half* d_outputImgArr = tensorcv::upload(256, 256);
            execTime[step][0] = duration_cast<nanoseconds>(high_resolution_clock::now()-blockStart).count();        

            tensorcv::synch();
            blockStart = high_resolution_clock::now();
            kernel.apply_resize(handle, d_inputImgArr, d_outputImgArr);
            execTime[step][1] = duration_cast<nanoseconds>(high_resolution_clock::now()-blockStart).count();
            tensorcv::synch();

            outputImg = tensorcv::download(d_outputImgArr, 256, 256);
            sprintf(filepath, "../img/output%d_resized.jpg", step%20+1);
            imwrite(filepath, outputImg);

            tensorcv::free(d_inputImgArr, d_outputImgArr);
            inputImg.release();
        }
        kernel.release_resize();

        kernel.init_crop(256, 256, 224, 224);
        kernel.upload_crop();
        for (int step=0; step<numImg; step++) {

            // load image from ../img/
            char filepath[32];
            sprintf(filepath, "../img/output%d_resized.jpg", step%20+1);
            Mat inputImg = imread(filepath, IMREAD_COLOR );
            if ( !inputImg.data ){ printf("No image data \n"); return -1;}
            half* d_inputImgArr = tensorcv::upload(&inputImg, inputImg.rows, inputImg.cols);
            half* d_outputImgArr = tensorcv::upload(224, 224);

            tensorcv::synch();
            auto blockStart = high_resolution_clock::now();
            kernel.apply_crop(handle, d_inputImgArr, d_outputImgArr);
            execTime[step][2] = duration_cast<nanoseconds>(high_resolution_clock::now()-blockStart).count();
            tensorcv::synch();

            outputImg = tensorcv::download(d_outputImgArr, 224, 224);
            sprintf(filepath, "../img/output%d_croped.jpg", step%20+1);
            imwrite(filepath, outputImg);

            tensorcv::free(d_inputImgArr, d_outputImgArr);
            inputImg.release();
        }
        kernel.release_crop();

        kernel.init_cvtcolor(224, 224, tensorcv::COLORCODE::RGB2YUV);
        kernel.upload_cvtcolor();
        for (int step=0; step<numImg; step++) {

            // load image from ../img/
            char filepath[32];
            sprintf(filepath, "../img/output%d_croped.jpg", step%20+1);
            Mat inputImg = imread(filepath, IMREAD_COLOR );
            if ( !inputImg.data ){ printf("No image data \n"); return -1;}
            half* d_inputImgArr = tensorcv::upload(&inputImg, inputImg.rows, inputImg.cols);
            half* d_outputImgArr = tensorcv::upload(224, 224);

            tensorcv::synch();
            auto blockStart = high_resolution_clock::now();
            kernel.apply_cvtcolor(handle, d_inputImgArr, d_outputImgArr);
            execTime[step][3] = duration_cast<nanoseconds>(high_resolution_clock::now()-blockStart).count();
            tensorcv::synch();

            outputImg = tensorcv::download(d_outputImgArr, 224, 224, 1);
            sprintf(filepath, "../img/output%d_cvtcolor.jpg", step%20+1);
            imwrite(filepath, outputImg);

            tensorcv::free(d_inputImgArr, d_outputImgArr);
            inputImg.release();
        }
        kernel.release_cvtcolor();

        tensorcv::splitted_src d_inputImgArr;
        tensorcv::splitted_src d_outputImgArr;

        kernel.init_rotate(224, 224, 1);
        kernel.upload_rotate();
        for (int step=0; step<numImg; step++) {

            // load image from ../img/
            char filepath[32];
            sprintf(filepath, "../img/output%d_croped.jpg", step%20+1);
            Mat inputImg = imread(filepath, IMREAD_COLOR );
            if ( !inputImg.data ){ printf("No image data \n"); return -1;}
            tensorcv::upload_split(&d_inputImgArr, &inputImg, inputImg.rows, inputImg.cols);
            tensorcv::upload_split(&d_outputImgArr, 224, 224);

            tensorcv::synch();
            auto blockStart = high_resolution_clock::now();
            kernel.apply_rotate(handle, d_inputImgArr.R, d_inputImgArr.G, d_inputImgArr.B, 
                d_outputImgArr.R, d_outputImgArr.G, d_outputImgArr.B);
            execTime[step][4] = duration_cast<nanoseconds>(high_resolution_clock::now()-blockStart).count();
            tensorcv::synch();

            outputImg = tensorcv::download_merge(d_outputImgArr.R, d_outputImgArr.G, d_outputImgArr.B, 224, 224);
            sprintf(filepath, "../img/output%d_rotate.jpg", step%20+1);
            imwrite(filepath, outputImg);

            tensorcv::free(d_inputImgArr, d_outputImgArr);
            inputImg.release();
        }
        kernel.release_rotate();
        
        kernel.init_normalize(224, 224, 1);
        kernel.upload_normalize();
        for (int step=0; step<numImg; step++) {

            // load image from ../img/
            char filepath[32];
            sprintf(filepath, "../img/output%d_croped.jpg", step%20+1);
            Mat inputImg = imread(filepath, IMREAD_COLOR );
            if ( !inputImg.data ){ printf("No image data \n"); return -1;}
            tensorcv::upload_split(&d_inputImgArr, &inputImg, inputImg.rows, inputImg.cols);
            tensorcv::upload_split(&d_outputImgArr, 224, 224);

            tensorcv::synch();
            auto blockStart = high_resolution_clock::now();
            kernel.apply_normalize(handle, d_inputImgArr.R, d_inputImgArr.G, d_inputImgArr.B, 
                d_outputImgArr.R, d_outputImgArr.G, d_outputImgArr.B);
            execTime[step][5] = duration_cast<nanoseconds>(high_resolution_clock::now()-blockStart).count();
            tensorcv::synch();

            outputImg = tensorcv::download_merge(d_outputImgArr.R, d_outputImgArr.G, d_outputImgArr.B, 224, 224);
            sprintf(filepath, "../img/output%d_normalize.jpg", step%20+1);
            imwrite(filepath, outputImg);

            tensorcv::free(d_inputImgArr, d_outputImgArr);
            inputImg.release();
        }
        kernel.release_normalize();

        tensorcv::imgprocKernel kernel_norm;

        kernel.init_integrated(4032, 3024, 256, 256, 224, 224, 1, 1);
        kernel.upload_integrated();
        for (int step=0; step<numImg; step++) {

            // load image from ../img/
            char filepath[32];
            sprintf(filepath, "../img/input%d.jpg", step%20+1);
            Mat inputImg = imread(filepath, IMREAD_COLOR );
            if ( !inputImg.data ){ printf("No image data \n"); return -1;}
            half* d_inputImgArr = tensorcv::upload(&inputImg, inputImg.rows, inputImg.cols);
            half* d_outputImgArr = tensorcv::upload(224, 224);

            tensorcv::synch();
            auto blockStart = high_resolution_clock::now();
            kernel.apply_integrated(handle, d_inputImgArr, d_outputImgArr);
            // TODO: rearrange the order of channels
            execTime[step][6] = duration_cast<nanoseconds>(high_resolution_clock::now()-blockStart).count();
            tensorcv::synch();

            outputImg = tensorcv::download(d_outputImgArr, 224, 224, 1);
            sprintf(filepath, "../img/output%d_integrated.jpg", step%20+1);
            imwrite(filepath, outputImg);

            tensorcv::free(d_inputImgArr, d_outputImgArr);
            inputImg.release();
        }
        kernel.release_integrated();
        
        outputImg.release();

        for (int step=0; step<numImg; step++) {
            std::cout << step << ",";
            for (int i=0; i<6; i++) {
                std::cout << execTime[step][i] << ",";
            }
            std::cout << execTime[step][6] << "\n";
        } 
    }

// ***************************************************************************************************
// Test functions
// ***************************************************************************************************

    // tensorcv::test();

    // // make input data
    // int iRow = 16, iCol = 32;

    // uchar* test = new uchar[iRow * iCol * 3];
    // for (int i = 0; i < iRow*iCol*3; i++)
    //     test[i] = rand() % 256;
    // Mat testImg(iRow, iCol, CV_MAKETYPE(CV_8U, 3), test);
    
    // for (int i = 0; i<iRow; i++) {
    //     for (int j=0; j<iCol*3; j++) {
    //         std::cout << (int)test[i*iCol*3+j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;

    // // upload and cudaMalloc
    // uchar* d_inputImgArr = tensorcv::upload(&testImg, testImg.rows, testImg.cols);
    // uchar* d_outputImgArr = tensorcv::upload(16, 32);
    
    // tensorcv::imgprocKernel kernel;

    // // kernel.init_resize(iRow, iCol, 16, 16);
    // // kernel.upload_resize();
    // // kernel.apply_resize(d_inputImgArr, d_outputImgArr);

    // // kernel.init_crop(iRow, iCol, 16, 16);
    // // kernel.upload_crop();
    // // kernel.apply_crop(d_inputImgArr, d_outputImgArr);

    // // kernel.init_cvtcolor(iRow, iCol, tensorcv::RGB2BGR);
    // // kernel.upload_cvtcolor();
    // // kernel.apply_cvtcolor(d_inputImgArr, d_outputImgArr);

    // // kernel.init_rotate(iRow, iCol, 2);
    // // kernel.upload_rotate();
    // // kernel.apply_rotate(d_inputImgArr, d_outputImgArr);

    // kernel.init_normalize(iRow, iCol, 1);
    // kernel.upload_normalize();
    // kernel.apply_normalize(d_inputImgArr, d_outputImgArr);

    // // kernel.init_integrated();
    // // kernel.upload_integrated();
    // // kernel.apply_integrated(d_inputImgArr, d_outputImgArr);

    // // download
    // Mat outputImg = tensorcv::download(d_outputImgArr, 16, 32);

    // for (int i = 0; i<outputImg.rows; i++) {
    //     for (int j=0; j<3*outputImg.cols; j++) {
    //         std::cout << (int)(outputImg.data[i*3*outputImg.cols+j]) << " ";
    //     }
    //     std::cout << std::endl;
    // }

    return 0;
}