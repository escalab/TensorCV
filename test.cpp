#include "test.hpp"

int main(int argc, char** argv )
{
    // check arguments
    if ( argc != 3 ) {
        printf("usage: ./tensorCV [CPU/GPU/CV/ALL] [inputSize]\n");
        printf("size: 480 / 1600 / 2048 / 2592 / 3264 / 4032\n");
        return -1;
    }

    // check GPUs
    std::cout << "We have " << cuda::getCudaEnabledDeviceCount() << " GPUs" << "\n";

    int long long execution_time = 0;
    int numImg = 20;

    // check mode
    // mode 0 : CPU 
    // mode 1 : GPU
    // mode 2 : TensorCV
    // mode 3 : all
    // mode 4 : test
    int mode;

    if(strcmp(argv[1],"CPU") == 0) mode = 0;
    else if(strcmp(argv[1],"GPU") == 0) mode = 1;
    else if(strcmp(argv[1],"CV") == 0) mode = 2;
    else if(strcmp(argv[1],"ALL") == 0) mode = 3;
    else if(strcmp(argv[1],"TEST") == 0) mode = 4;
    else {
        printf("usage: ./tensorCV [CPU/GPU/CV/ALL]\n");
        printf("size: 480 / 1600 / 2048 / 2592 / 3264 / 4032\n");
        return -1;
    }

    // check input size
    int inputSize = atoi(argv[2]);
    if (inputSize != 480 && inputSize != 1600 && inputSize != 2048 
        && inputSize != 2592 && inputSize != 3264 && inputSize !=4032) {
        printf("usage: ./tensorCV [CPU/GPU/CV/ALL] [inputSize]\n");
        printf("size: 480 / 1600 / 2048 / 2592 / 3264 / 4032\n");
        return -1;
    }
    
// ***************************************************************************************************

    std::cout << "OpenCV img preprocessing with CPU" << std::endl;
    std::cout << "Iter resiz center cvt rotate norm" << std::endl;

    // declare variables
    Mat outputImg, smallImg, outputImg_rgb[3];

    if (mode == 0 || mode == 3) {
        for (int step=0; step<numImg; step++) {
            std::cout << step << " ";

            // load image from ../img/inputSize/
            char filepath[32] = {};
            sprintf(filepath, "../img/%d/input%d.jpg", inputSize, step+1);
            Mat inputImg = imread(filepath, IMREAD_COLOR );
            if ( !inputImg.data ){ printf("No image data \n"); return -1;}
        
            // resize
            auto blockStart = high_resolution_clock::now();
            Size resizeSize(256,256);
            resize(inputImg, smallImg, resizeSize, 0, 0, INTER_LINEAR);
            std::cout << duration_cast<nanoseconds>(high_resolution_clock::now()-blockStart).count() << " ";

            // center crop
            blockStart = high_resolution_clock::now();
            Rect rect(16,16,224,224);
            outputImg = smallImg(rect);
            std::cout << duration_cast<nanoseconds>(high_resolution_clock::now()-blockStart).count() << " ";

            // cvt color
            blockStart = high_resolution_clock::now();
            // COLOR_BGR2GRAY, COLOR_BGR2RGB, COLOR_RGB2YCrCb, ...
            cvtColor(outputImg, outputImg, COLOR_RGB2YCrCb, 0);
            std::cout << duration_cast<nanoseconds>(high_resolution_clock::now()-blockStart).count() << " ";

            // rotate
            blockStart = high_resolution_clock::now();
            rotate(outputImg, outputImg, ROTATE_90_CLOCKWISE);
            std::cout << duration_cast<nanoseconds>(high_resolution_clock::now()-blockStart).count() << " ";
            
            // normalize
            blockStart = high_resolution_clock::now();
            split(outputImg, outputImg_rgb);
            normalize(outputImg_rgb[0], outputImg_rgb[0], 1.0, 0.0, NORM_L2, CV_32F);
            normalize(outputImg_rgb[1], outputImg_rgb[1], 1.0, 0.0, NORM_L2, CV_32F);
            normalize(outputImg_rgb[2], outputImg_rgb[2], 1.0, 0.0, NORM_L2, CV_32F);
            merge(outputImg_rgb, 3, outputImg);
            std::cout << duration_cast<nanoseconds>(high_resolution_clock::now()-blockStart).count() << "\n";

            inputImg.release();

            // imwrite("output.jpg", outputImg);
        }
    }

// ***************************************************************************************************

    std::cout << "OpenCV img preprocessing with GPU" << std::endl;
    std::cout << "Iter resiz center cvt rotate norm" << std::endl;

    cuda::GpuMat inputImg_gpu, outputImg_gpu, outputImg_gpu_rgb[3];

    if (mode == 1 || mode == 3) {
        for (int step=0; step<numImg; step++) {
            std::cout << step << " ";

            // load image from ../img/
            char filepath[32];
            sprintf(filepath, "../img/%d/input%d.jpg", inputSize, step+1);
            Mat inputImg = imread(filepath, IMREAD_COLOR );
            if ( !inputImg.data ){ printf("No image data \n"); return -1;}
            inputImg_gpu.upload(inputImg);

            // resize
            auto blockStart = high_resolution_clock::now();
            Size resizeSize(256,256);
            cuda::resize(inputImg_gpu, outputImg_gpu, resizeSize, 0, 0, INTER_LINEAR);
            std::cout << duration_cast<nanoseconds>(high_resolution_clock::now()-blockStart).count() << " ";

            // center crop
            blockStart = high_resolution_clock::now();
            Rect rect(16,16,224,224);
            outputImg_gpu = outputImg_gpu(rect);
            std::cout << duration_cast<nanoseconds>(high_resolution_clock::now()-blockStart).count() << " ";
        
            // cvt color
            blockStart = high_resolution_clock::now();
            // COLOR_BGR2GRAY, COLOR_BGR2RGB, COLOR_RGB2YCrCb, ...
            cuda::cvtColor(outputImg_gpu, outputImg_gpu, COLOR_RGB2YCrCb, 0);
            std::cout << duration_cast<nanoseconds>(high_resolution_clock::now()-blockStart).count() << " ";

            // rotate
            blockStart = high_resolution_clock::now();
            Size rotateSize(224,224);
            cuda::rotate(outputImg_gpu, outputImg_gpu, rotateSize, 90, 0, 224);
            std::cout << duration_cast<nanoseconds>(high_resolution_clock::now()-blockStart).count() << " ";

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

            // imwrite("output.jpg", outputImg);
        }
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

    std::cout << "Img preprocessing with TensorCV" << std::endl;
    std::cout << "Iter resiz center cvt rotate norm integ" << std::endl;
    long int execTime[numImg][7] = {0};

    if (mode == 2 || mode == 3) {
        // generate resize kernels
        switch (inputSize)
        {
            case 4032:
                kernel.init_resize(4032, 3024, 256, 256);
                break;
            case 3264:
                kernel.init_resize(3264, 2448, 256, 256);
                break;
            case 2592:
                kernel.init_resize(2592, 1936, 256, 256);
                break;
            case 2048:
                kernel.init_resize(2048, 1536, 256, 256);
                break; 
            case 1600:
                kernel.init_resize(1600, 1200, 256, 256);
                break; 
            case 480:
                kernel.init_resize(480, 320, 256, 256);
                break;
            default:
                printf("input size error\n");
                break;
        }
        kernel.upload_resize();
        for (int step=0; step<numImg; step++) {

            // load image from ../img/
            char filepath[32];
            sprintf(filepath, "../img/%d/input%d.jpg", inputSize, step+1);
            Mat inputImg = imread(filepath, IMREAD_COLOR );
            if ( !inputImg.data ){ printf("No image data \n"); return -1;}
            // upload and cudaMalloc
            half* d_inputImgArr = tensorcv::upload(&inputImg, inputImg.rows, inputImg.cols);
            half* d_outputImgArr = tensorcv::upload(256, 256);

            // apply resize kernel
            auto blockStart = high_resolution_clock::now();
            kernel.apply_resize(handle, d_inputImgArr, d_outputImgArr);
            execTime[step][1] = duration_cast<nanoseconds>(high_resolution_clock::now()-blockStart).count();

            // download and free
            outputImg = tensorcv::download(d_outputImgArr, 256, 256);
            sprintf(filepath, "../img/output%d_resized.jpg", step+1);
            imwrite(filepath, outputImg);
            tensorcv::free(d_inputImgArr, d_outputImgArr);
            inputImg.release();
        }
        kernel.release_resize();

        // generate crop kernels
        kernel.init_crop(256, 256, 224, 224);
        kernel.upload_crop();
        for (int step=0; step<numImg; step++) {

            // load image from ../img/
            char filepath[32];
            sprintf(filepath, "../img/output%d_resized.jpg", step+1);
            Mat inputImg = imread(filepath, IMREAD_COLOR );
            if ( !inputImg.data ){ printf("No image data \n"); return -1;}
            // upload and cudaMalloc
            half* d_inputImgArr = tensorcv::upload(&inputImg, inputImg.rows, inputImg.cols);
            half* d_outputImgArr = tensorcv::upload(224, 224);

            // apply resize kernel
            auto blockStart = high_resolution_clock::now();
            kernel.apply_crop(handle, d_inputImgArr, d_outputImgArr);
            execTime[step][2] = duration_cast<nanoseconds>(high_resolution_clock::now()-blockStart).count();

            // download and free
            outputImg = tensorcv::download(d_outputImgArr, 224, 224);
            sprintf(filepath, "../img/output%d_croped.jpg", step+1);
            imwrite(filepath, outputImg);
            tensorcv::free(d_inputImgArr, d_outputImgArr);
            inputImg.release();
        }
        kernel.release_crop();

        // generate cvtcolor kernels
        kernel.init_cvtcolor(224, 224, tensorcv::COLORCODE::RGB2YUV);
        kernel.upload_cvtcolor();
        for (int step=0; step<numImg; step++) {

            // load image from ../img/
            char filepath[32];
            sprintf(filepath, "../img/output%d_croped.jpg", step+1);
            Mat inputImg = imread(filepath, IMREAD_COLOR );
            if ( !inputImg.data ){ printf("No image data \n"); return -1;}
            // upload and cudaMalloc
            half* d_inputImgArr = tensorcv::upload(&inputImg, inputImg.rows, inputImg.cols);
            half* d_outputImgArr = tensorcv::upload(224, 224);

            // apply resize kernel
            auto blockStart = high_resolution_clock::now();
            kernel.apply_cvtcolor(handle, d_inputImgArr, d_outputImgArr);
            execTime[step][3] = duration_cast<nanoseconds>(high_resolution_clock::now()-blockStart).count();

            // download and free
            outputImg = tensorcv::download(d_outputImgArr, 224, 224, 1);
            sprintf(filepath, "../img/output%d_cvtcolor.jpg", step+1);
            imwrite(filepath, outputImg);
            tensorcv::free(d_inputImgArr, d_outputImgArr);
            inputImg.release();
        }
        kernel.release_cvtcolor();

        tensorcv::splitted_src d_inputImgArr;
        tensorcv::splitted_src d_outputImgArr;

        // generate rotate kernels
        kernel.init_rotate(224, 224, 1);
        kernel.upload_rotate();
        for (int step=0; step<numImg; step++) {

            // load image from ../img/
            char filepath[32];
            sprintf(filepath, "../img/output%d_croped.jpg", step+1);
            Mat inputImg = imread(filepath, IMREAD_COLOR );
            if ( !inputImg.data ){ printf("No image data \n"); return -1;}
            // upload and cudaMalloc
            tensorcv::upload_split(&d_inputImgArr, &inputImg, inputImg.rows, inputImg.cols);
            tensorcv::upload_split(&d_outputImgArr, 224, 224);

            // apply resize kernel
            auto blockStart = high_resolution_clock::now();
            kernel.apply_rotate(handle, d_inputImgArr.R, d_inputImgArr.G, d_inputImgArr.B, 
                d_outputImgArr.R, d_outputImgArr.G, d_outputImgArr.B);
            execTime[step][4] = duration_cast<nanoseconds>(high_resolution_clock::now()-blockStart).count();

            // download and free
            outputImg = tensorcv::download_merge(d_outputImgArr.R, d_outputImgArr.G, d_outputImgArr.B, 224, 224);
            sprintf(filepath, "../img/output%d_rotate.jpg", step+1);
            imwrite(filepath, outputImg);
            tensorcv::free(d_inputImgArr, d_outputImgArr);
            inputImg.release();
        }
        kernel.release_rotate();
        
        // generate normalize kernels
        kernel.init_normalize(224, 224, 1);
        kernel.upload_normalize();
        for (int step=0; step<numImg; step++) {

            // load image from ../img/
            char filepath[32];
            sprintf(filepath, "../img/output%d_croped.jpg", step+1);
            Mat inputImg = imread(filepath, IMREAD_COLOR );
            if ( !inputImg.data ){ printf("No image data \n"); return -1;}
            // upload and cudaMalloc
            tensorcv::upload_split(&d_inputImgArr, &inputImg, inputImg.rows, inputImg.cols);
            tensorcv::upload_split(&d_outputImgArr, 224, 224);

            // apply resize kernel
            auto blockStart = high_resolution_clock::now();
            kernel.apply_normalize(handle, d_inputImgArr.R, d_inputImgArr.G, d_inputImgArr.B, 
                d_outputImgArr.R, d_outputImgArr.G, d_outputImgArr.B);
            execTime[step][5] = duration_cast<nanoseconds>(high_resolution_clock::now()-blockStart).count();

            // download and free
            outputImg = tensorcv::download_merge(d_outputImgArr.R, d_outputImgArr.G, d_outputImgArr.B, 224, 224);
            sprintf(filepath, "../img/output%d_normalize.jpg", step+1);
            imwrite(filepath, outputImg);
            tensorcv::free(d_inputImgArr, d_outputImgArr);
            inputImg.release();
        }
        kernel.release_normalize();

        tensorcv::imgprocKernel kernel_norm;

        // generate integrated kernels
        switch (inputSize)
        {
            case 4032:
                kernel.init_integrated(4032, 3024, 256, 256, 224, 224, 1, 1);
                break;
            case 3264:
                kernel.init_integrated(3264, 2448, 256, 256, 224, 224, 1, 1);
                break;
            case 2592:
                kernel.init_integrated(2592, 1936, 256, 256, 224, 224, 1, 1);
                break;
            case 2048:
                kernel.init_integrated(2048, 1536, 256, 256, 224, 224, 1, 1);
                break; 
            case 1600:
                kernel.init_integrated(1600, 1200, 256, 256, 224, 224, 1, 1);
                break; 
            case 480:
                kernel.init_integrated(480, 320, 256, 256, 224, 224, 1, 1);
                break;
            default:
                printf("input size error\n");
                break;
        }
        kernel.upload_integrated();
        for (int step=0; step<numImg; step++) {

            // load image from ../img/
            char filepath[32];
            sprintf(filepath, "../img/%d/input%d.jpg", inputSize, step+1);
            Mat inputImg = imread(filepath, IMREAD_COLOR );
            if ( !inputImg.data ){ printf("No image data \n"); return -1;}
            half* d_inputImgArr = tensorcv::upload(&inputImg, inputImg.rows, inputImg.cols);
            half* d_outputImgArr = tensorcv::upload(224, 224);

            auto blockStart = high_resolution_clock::now();
            kernel.apply_integrated(handle, d_inputImgArr, d_outputImgArr);
            // TODO: rearrange the order of channels
            execTime[step][6] = duration_cast<nanoseconds>(high_resolution_clock::now()-blockStart).count();

            outputImg = tensorcv::download(d_outputImgArr, 224, 224, 1);
            sprintf(filepath, "../img/output%d_integrated.jpg", step+1);
            imwrite(filepath, outputImg);

            tensorcv::free(d_inputImgArr, d_outputImgArr);
            inputImg.release();
        }
        kernel.release_integrated();
        
        outputImg.release();

        for (int step=0; step<numImg; step++) {
            std::cout << step << " ";
            for (int i=1; i<6; i++) {
                std::cout << execTime[step][i] << " ";
            }
            std::cout << execTime[step][6] << "\n";
        } 
    }

// ***************************************************************************************************
// Test functions
// ***************************************************************************************************

    if (mode == 4) {
        // tensorcv::GEMMtest();

        // make input data
        int iRow = 4, iCol = 4;

        uchar* test = new uchar[iRow * iCol * 3];
        for (int i = 0; i < iRow*iCol*3; i++)
            test[i] = rand() % 255;
        Mat testImg(iRow, iCol, CV_MAKETYPE(CV_8U, 3), test);
        
        for (int i = 0; i<iRow; i++) {
            for (int j=0; j<iCol*3; j++) {
                std::cout << (int)test[i*iCol*3+j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;

        // upload and cudaMalloc
        // uchar* d_inputImgArr = tensorcv::upload(&testImg, testImg.rows, testImg.cols);
        // uchar* d_outputImgArr = tensorcv::upload(16, 32);
        
        // tensorcv::imgprocKernel kernel;

        // kernel.init_resize(iRow, iCol, 16, 16);
        // kernel.upload_resize();
        // kernel.apply_resize(d_inputImgArr, d_outputImgArr);

        // kernel.init_crop(iRow, iCol, 16, 16);
        // kernel.upload_crop();
        // kernel.apply_crop(d_inputImgArr, d_outputImgArr);

        // kernel.init_cvtcolor(iRow, iCol, tensorcv::RGB2BGR);
        // kernel.upload_cvtcolor();
        // kernel.apply_cvtcolor(d_inputImgArr, d_outputImgArr);

        // kernel.init_rotate(iRow, iCol, 2);
        // kernel.upload_rotate();
        // kernel.apply_rotate(d_inputImgArr, d_outputImgArr);

        // kernel.init_normalize(iRow, iCol, 1);
        // kernel.upload_normalize();
        // kernel.apply_normalize(d_inputImgArr, d_outputImgArr);

        // kernel.init_integrated();
        // kernel.upload_integrated();
        // kernel.apply_integrated(d_inputImgArr, d_outputImgArr);

// ***************************************************************************************************
        // Normalize test

        kernel.init_normalize(4, 4, 1);
        kernel.upload_normalize();

        tensorcv::splitted_src d_inputImgArr;
        tensorcv::splitted_src d_outputImgArr;

        // load image from ../img/
        tensorcv::upload_split(&d_inputImgArr, &testImg, testImg.rows, testImg.cols);
        tensorcv::upload_split(&d_outputImgArr, 4, 4);

        kernel.apply_normalize(handle, d_inputImgArr.R, d_inputImgArr.G, d_inputImgArr.B, 
            d_outputImgArr.R, d_outputImgArr.G, d_outputImgArr.B);

        outputImg = tensorcv::download_merge(d_outputImgArr.R, d_outputImgArr.G, d_outputImgArr.B, 4, 4);
        for (int i = 0; i<outputImg.rows; i++) {
            for (int j=0; j<3*outputImg.cols; j++) {
                std::cout << (int)(outputImg.data[i*3*outputImg.cols+j]) << " ";
            }
            std::cout << std::endl;
        }
    }
    return 0;
}