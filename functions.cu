#include "functions.hpp"

// Detect errors
#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
    if (stat != cudaSuccess)
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
}
#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
    if (stat != CUBLAS_STATUS_SUCCESS)
        fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
}

void tensorcv::free (half* input, half* output) {
    cudaFree(input);
    cudaFree(output);
}

void tensorcv::free (tensorcv::splitted_src input, tensorcv::splitted_src output) {
    cudaFree(input.R);
    cudaFree(input.G);
    cudaFree(input.B);
    cudaFree(output.R);
    cudaFree(output.G);
    cudaFree(output.B);
}

// ****************************************************************************************************
// Type conversion
// ****************************************************************************************************

__global__ void uchar2half(half* dst, unsigned char* src, int rows, int cols, int transpose, int alpha=255){
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx >= cols || ty >= rows) return;
    if (transpose)
        dst[tx*rows + ty] = (half) ((float)src[ty*cols + tx] / alpha);
    else
        dst[ty*cols + tx] = (half) ((float)src[ty*cols + tx] / alpha);
}

__global__ void half2uchar(unsigned char* dst, half* src, int rows, int cols, int transpose, int alpha=255){
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx >= cols || ty >= rows) return;
    if (transpose == 1) {
        dst[ty*cols + tx] = (unsigned char)((float)src[tx*rows + ty] * alpha);
    } else if (transpose == 2) {
        dst[ty*cols + tx] = (unsigned char)((float)src[(tx/3)*cols + (3*ty+tx%3)] * alpha);
    } else {
        dst[ty*cols + tx] = (unsigned char)((float)src[ty*cols + tx] * alpha);
    }
}

__global__ void uchar2half_split(half* dst1, half* dst2, half* dst3, unsigned char* src, int rows, int cols, int transpose, int alpha=255){
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx >= cols || ty >= rows) return;
    if (transpose) {
        dst1[tx*rows + ty] = (half) ((float)src[ty*3*cols + 3*tx] / alpha);
        dst2[tx*rows + ty] = (half) ((float)src[ty*3*cols + 3*tx+1] / alpha);
        dst3[tx*rows + ty] = (half) ((float)src[ty*3*cols + 3*tx+2] / alpha);
    } else {
        dst1[ty*cols + tx] = (half) ((float)src[ty*3*cols + 3*tx] / alpha);
        dst2[ty*cols + tx] = (half) ((float)src[ty*3*cols + 3*tx+1] / alpha);
        dst3[ty*cols + tx] = (half) ((float)src[ty*3*cols + 3*tx+2] / alpha);
    }
}

__global__ void half2uchar_merge(unsigned char* dst, half* src1, half* src2, half* src3, int rows, int cols, int transpose, int alpha=255){
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx >= cols || ty >= rows) return;
    if (transpose) {
        dst[ty*3*cols + 3*tx] = (unsigned char)((float)src1[tx*rows + ty] * alpha);
        dst[ty*3*cols + 3*tx+1] = (unsigned char)((float)src2[tx*rows + ty] * alpha);
        dst[ty*3*cols + 3*tx+2] = (unsigned char)((float)src3[tx*rows + ty] * alpha);
    } else {
        dst[ty*3*cols + 3*tx] = (unsigned char)((float)src1[ty*cols + tx] * alpha);
        dst[ty*3*cols + 3*tx+1] = (unsigned char)((float)src2[ty*cols + tx] * alpha);
        dst[ty*3*cols + 3*tx+2] = (unsigned char)((float)src3[ty*cols + tx] * alpha);
    }
}

// ****************************************************************************************************
// Upload and download data form GPUs
// ****************************************************************************************************

half* tensorcv::upload ( Mat* src, int rows, int cols ){
    unsigned char* src_d = NULL;
    cudaErrCheck( cudaMalloc((void **)&src_d, rows * cols * 3 * sizeof(unsigned char)) );
    cudaErrCheck( cudaMemcpy(src_d, src->data, rows * cols * 3, cudaMemcpyHostToDevice) );
    
    half* input_d = NULL;
    cudaErrCheck( cudaMalloc((void **)&input_d, rows * cols * 3 * sizeof(half)) );
    cudaErrCheck( cudaMemset(input_d, 0, rows * cols * 3 * sizeof(half)));
    
    dim3 block(32, 32);
    dim3 grid((3*cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    uchar2half<<<grid, block>>>(input_d, src_d, rows, 3*cols, 0);

    cudaDeviceSynchronize();
    cudaErrCheck( cudaFree(src_d) );
    return input_d;
}

half* tensorcv::upload ( int rows, int cols ){
    half* input_d = NULL;
    cudaErrCheck( cudaMalloc((void **)&input_d, rows * cols * 3 * sizeof(half)) );
    cudaErrCheck( cudaMemset(input_d, 0, rows * cols * 3 * sizeof(half)));
    cudaDeviceSynchronize();
    return input_d;
}

Mat tensorcv::download ( half* output_d, int rows, int cols, int transpose ){
    unsigned char* dst_d = NULL;
    cudaErrCheck( cudaMalloc((void **)&dst_d, rows * cols * 3 * sizeof(unsigned char)) );
    cudaErrCheck( cudaMemset(dst_d, 0, rows * cols * 3 * sizeof(unsigned char)) );

    dim3 block(16, 16);
    dim3 grid((3*cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    half2uchar<<<grid, block>>>(dst_d, output_d, rows, 3*cols, transpose);
    
    unsigned char* download_output = new unsigned char[rows * cols * 3];
    cudaErrCheck( cudaMemcpy(download_output, dst_d, rows * cols * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost) );

    cudaErrCheck( cudaFree(dst_d) );
    return Mat(rows, cols, CV_MAKETYPE(CV_8U, 3), download_output);
}

void tensorcv::upload_split ( tensorcv::splitted_src* dst, Mat* src, int rows, int cols ){
    unsigned char* src_d = NULL;
    cudaErrCheck( cudaMalloc((void **)&src_d, rows * cols * 3 * sizeof(unsigned char)) );
    cudaErrCheck( cudaMemcpy(src_d, src->data, rows * cols * 3, cudaMemcpyHostToDevice) );
    
    cudaErrCheck( cudaMalloc((void **)&(dst->R), rows * cols * sizeof(half)) );
    cudaErrCheck( cudaMemset((dst->R), 0, rows * cols * sizeof(half)));
    cudaErrCheck( cudaMalloc((void **)&(dst->G), rows * cols * sizeof(half)) );
    cudaErrCheck( cudaMemset((dst->G), 0, rows * cols * sizeof(half)));
    cudaErrCheck( cudaMalloc((void **)&(dst->B), rows * cols * sizeof(half)) );
    cudaErrCheck( cudaMemset((dst->B), 0, rows * cols * sizeof(half)));
    
    dim3 block(16, 16);
    dim3 grid((3*cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    uchar2half_split<<<grid, block>>>((dst->R), (dst->G), (dst->B), src_d, rows, cols, 0);

    cudaDeviceSynchronize();
    cudaErrCheck( cudaFree(src_d) );
}

// TODO: memset 1
void tensorcv::upload_split ( tensorcv::splitted_src* dst, int rows, int cols ){
    cudaErrCheck( cudaMalloc((void **)&(dst->R), rows * cols * sizeof(half)) );
    cudaErrCheck( cudaMemset((dst->R), 0, rows * cols * sizeof(half)));
    cudaErrCheck( cudaMalloc((void **)&(dst->G), rows * cols * sizeof(half)) );
    cudaErrCheck( cudaMemset((dst->G), 0, rows * cols * sizeof(half)));
    cudaErrCheck( cudaMalloc((void **)&(dst->B), rows * cols * sizeof(half)) );
    cudaErrCheck( cudaMemset((dst->B), 0, rows * cols * sizeof(half)));
    cudaDeviceSynchronize();
}

Mat tensorcv::download_merge ( half* output1_d, half* output2_d, half* output3_d, int rows, int cols, int transpose ){
    unsigned char* dst_d = NULL;
    cudaErrCheck( cudaMalloc((void **)&dst_d, rows * cols * 3 * sizeof(unsigned char)) );
    cudaErrCheck( cudaMemset(dst_d, 0, rows * cols * 3 * sizeof(unsigned char)) );

    dim3 block(16, 16);
    dim3 grid((3*cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    half2uchar_merge<<<grid, block>>>(dst_d, output1_d, output2_d, output3_d, rows, 3*cols, transpose);

    unsigned char* download_output = new unsigned char[rows * cols * 3];
    cudaErrCheck( cudaMemcpy(download_output, dst_d, rows * cols * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost) );

    cudaErrCheck( cudaFree(dst_d) );
    return Mat(rows, cols, CV_MAKETYPE(CV_8U, 3), download_output);
}


// ****************************************************************************************************
// Resize
// ****************************************************************************************************

void tensorcv::imgprocKernel::init_resize(int iRow_, int iCol_, int oRow_, int oCol_){
    iRow = iRow_; iCol = iCol_; oRow = oRow_; oCol = oCol_;
    rowFactor = (float)oRow / (float)iRow;
    colFactor = (float)oCol / (float)iCol;

    kernel1 = new half[oRow * iRow]();
    kernel2 = new half[3*oCol * 3*iCol]();

    for (int i=0; i<oRow; i++) {
        int top = floor(i/rowFactor);
        int bot = ceil(i/rowFactor);
        float rowWeight = (float)i/rowFactor - top;
        if (rowWeight == 0) {
            kernel1[i*iRow + top] = 1;
        } else {
            kernel1[i*iRow + top] = (float)(1 - rowWeight);
            kernel1[i*iRow + bot] = (float)(rowWeight);    
        }
    }
    for (int j=0; j<oCol; j++) {    
        int left = floor(j/colFactor);
        int right = ceil(j/colFactor);
        float colWeight = (float)j/colFactor - left;
        if (colWeight == 0) {
            kernel2[(3*j)*3*iCol + 3*left] = 1;
            kernel2[(3*j+1)*3*iCol + 3*left+1] = 1;
            kernel2[(3*j+2)*3*iCol + 3*left+2] = 1;
        } else {
            kernel2[(3*j)*3*iCol + 3*left] = (half)(1 - colWeight);
            kernel2[(3*j+1)*3*iCol + 3*left+1] = (half)(1 - colWeight);
            kernel2[(3*j+2)*3*iCol + 3*left+2] = (half)(1 - colWeight);
            kernel2[(3*j)*3*iCol + 3*right] = (half)(colWeight);
            kernel2[(3*j+1)*3*iCol + 3*right+1] = (half)(colWeight);
            kernel2[(3*j+2)*3*iCol + 3*right+2] = (half)(colWeight);
        }
    }
}

void tensorcv::imgprocKernel::upload_resize() {
    cudaErrCheck( cudaMalloc((void **)&d_kernel1, oRow * iRow * sizeof(half)) );
    cudaErrCheck( cudaMemcpy(d_kernel1, kernel1, oRow * iRow  * sizeof(half), cudaMemcpyHostToDevice) );
    cudaErrCheck( cudaMalloc((void **)&d_kernel2, iCol*3 * oCol*3 * sizeof(half)) );
    cudaErrCheck( cudaMemcpy(d_kernel2, kernel2, iCol*3 * oCol*3  * sizeof(half), cudaMemcpyHostToDevice) );
    cudaErrCheck( cudaMalloc((void **)&d_temp1, oRow * 3 * iCol * sizeof(half)) );
}

void tensorcv::imgprocKernel::apply_resize( cublasHandle_t handle, half* src, half* dst ) {
    const half alpha = 1.0;
    const half beta = 0.0;
    cublasErrCheck( cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, oRow, 3*iCol, iRow,
                                    &alpha, d_kernel1, CUDA_R_16F, iRow, src, CUDA_R_16F, 3*iCol, 
                                    &beta, d_temp1, CUDA_R_16F, oRow,
                                    CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) );
    cublasErrCheck( cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, 3*oCol, oRow, 3*iCol, 
                                    &alpha, d_kernel2, CUDA_R_16F, 3*iCol, d_temp1, CUDA_R_16F, oRow, 
                                    &beta, dst, CUDA_R_16F, 3*oCol, 
                                    CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) );
}

void tensorcv::imgprocKernel::release_resize() {
    cudaFree(d_kernel1);
    cudaFree(d_kernel2);
    cudaFree(d_temp1);
    delete[] kernel1;
    delete[] kernel2;
}

// ****************************************************************************************************
// Crop
// ****************************************************************************************************

void tensorcv::imgprocKernel::init_crop(int iRow_, int iCol_, int oRow_, int oCol_){
    iRow = iRow_; iCol = iCol_; oRow = oRow_; oCol = oCol_;
    rowFactor = iRow/2 - oRow/2;
    colFactor = iCol/2 - oCol/2;

    kernel1 = new half[oRow * iRow]();
    kernel2 = new half[3*oCol * 3*iCol]();

    for (int i=0; i<oRow; i++) {
        kernel1[i*iRow + i + (int)rowFactor] = 1;
    }
    for (int j=0; j<oCol; j++) {    
        kernel2[(3*j)*3*iCol + 3*j + 3*(int)colFactor] = 1;
        kernel2[(3*j+1)*3*iCol + 3*j + 1 + 3*(int)colFactor] = 1;
        kernel2[(3*j+2)*3*iCol + 3*j + 2 + 3*(int)colFactor] = 1;
    }
}

void tensorcv::imgprocKernel::upload_crop() {
    cudaErrCheck( cudaMalloc((void **)&d_kernel1, oRow * iRow * sizeof(half)) );
    cudaErrCheck( cudaMemcpy(d_kernel1, kernel1, oRow * iRow  * sizeof(half), cudaMemcpyHostToDevice) );
    cudaErrCheck( cudaMalloc((void **)&d_kernel2, iCol*3 * oCol*3 * sizeof(half)) );
    cudaErrCheck( cudaMemcpy(d_kernel2, kernel2, iCol*3 * oCol*3  * sizeof(half), cudaMemcpyHostToDevice) );
    cudaErrCheck( cudaMalloc((void **)&d_temp1, oRow * 3 * iCol * sizeof(half)) );
}

void tensorcv::imgprocKernel::apply_crop( cublasHandle_t handle, half* src, half* dst ) {
    const half alpha = 1.0;
    const half beta = 0.0;

    cublasErrCheck( cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, oRow, 3*iCol, iRow,
                                    &alpha, d_kernel1, CUDA_R_16F, iRow, src, CUDA_R_16F, 3*iCol, 
                                    &beta, d_temp1, CUDA_R_16F, oRow,
                                    CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) );
    cublasErrCheck( cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, 3*oCol, oRow, 3*iCol, 
                                    &alpha, d_kernel2, CUDA_R_16F, 3*iCol, d_temp1, CUDA_R_16F, oRow, 
                                    &beta, dst, CUDA_R_16F, 3*oCol, 
                                    CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) );
}

void tensorcv::imgprocKernel::release_crop() {
    cudaFree(d_kernel1);
    cudaFree(d_kernel2);
    cudaFree(d_temp1);
    delete[] kernel1;
    delete[] kernel2;
}

// ****************************************************************************************************
// CvtColor
// ****************************************************************************************************

half colorPallet[6][9] = {{0,0,1,0,1,0,1,0,0},      // RGB2BGR
                          {0,0,1,0,1,0,1,0,0},      // BGR2RGB
                          {0.299,-0.14713,0.615,
                           0.587,-0.28886,-0.51499,
                           0.114,0.436,-0.10001},   // RGB2YUV
                          {1,1,1,
                           0,-0.39465,2.03211,
                           1.13983,-0.58060,0},     // YUV2RGB
                          {0.114,0.436,-0.10001,
                           0.587,-0.28886,-0.51499,
                           0.299,-0.14713,0.615},   // BGR2YUV
                          {1.13983,-0.58060,0,
                           0,-0.39465,2.03211,
                           1,1,1}};                 // YUV2BGR

void tensorcv::imgprocKernel::init_cvtcolor(int iRow_, int iCol_, int colorCode_){
    iRow = iRow_; iCol = iCol_;
    colorCode = colorCode_;

    kernel1 = new half[3*iCol*3*iCol]();

    for (int i=0; i<iCol; i++) {
        kernel1[(3*i)*3*iCol + 3*i] = colorPallet[colorCode][0];
        kernel1[(3*i)*3*iCol + 3*i+1] = colorPallet[colorCode][1];
        kernel1[(3*i)*3*iCol + 3*i+2] = colorPallet[colorCode][2];
        kernel1[(3*i+1)*3*iCol + 3*i] = colorPallet[colorCode][3];
        kernel1[(3*i+1)*3*iCol + 3*i+1] = colorPallet[colorCode][4];
        kernel1[(3*i+1)*3*iCol + 3*i+2] = colorPallet[colorCode][5];
        kernel1[(3*i+2)*3*iCol + 3*i] = colorPallet[colorCode][6];
        kernel1[(3*i+2)*3*iCol + 3*i+1] = colorPallet[colorCode][7];
        kernel1[(3*i+2)*3*iCol + 3*i+2] = colorPallet[colorCode][8];
    }
}

void tensorcv::imgprocKernel::upload_cvtcolor() {
    cudaErrCheck( cudaMalloc((void **)&d_kernel1, 3*iCol*3*iCol * sizeof(half)) );
    cudaErrCheck( cudaMemcpy(d_kernel1, kernel1, 3*iCol*3*iCol * sizeof(half), cudaMemcpyHostToDevice) );
    cudaErrCheck( cudaMalloc((void **)&d_temp1, iRow * 3 * iCol * sizeof(half)) );
    // cudaErrCheck( cudaMemset(d_temp1, 0, iRow * 3 * iCol * sizeof(half)) );
}

void tensorcv::imgprocKernel::apply_cvtcolor( cublasHandle_t handle, half* src, half* dst ) {
    const half alpha = 1.0;
    const half beta = 0.0;

    cublasErrCheck( cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, iRow, 3*iCol, 3*iCol,
                                    &alpha, src, CUDA_R_16F, 3*iCol, d_kernel1, CUDA_R_16F, 3*iCol, 
                                    &beta, dst, CUDA_R_16F, iRow,
                                    CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) );
}

void tensorcv::imgprocKernel::release_cvtcolor() {
    cudaFree(d_kernel1);
    cudaFree(d_temp1);
    delete[] kernel1;
}

// ****************************************************************************************************
// Rotate
// ****************************************************************************************************

void tensorcv::imgprocKernel::init_rotate(int iRow_, int iCol_, int repeat_){
    iRow = iRow_; 
    iCol = iCol_;
    repeat = repeat_;

    if (repeat % 4 == 1) {
        // 90-degree rotation
        kernel1 = new half[iCol*iCol]();
        for (int i=0; i<iCol; i++)
            kernel1[i*iCol + (iCol-1-i)] = 1;

    } else if (repeat % 4 == 2) {
        // 180-degree rotation
        kernel1 = new half[iCol*iCol]();
        kernel2 = new half[iRow*iRow]();
        for (int i=0; i<iCol; i++)
            kernel1[i*iCol + (iCol-1-i)] = 1;
        for (int i=0; i<iRow; i++)
            kernel2[i*iRow + (iRow-1-i)] = 1;
        
    } else if (repeat % 4 == 3) {
        // 270-degree rotation
        kernel2 = new half[iRow*iRow]();
        for (int i=0; i<iRow; i++)
            kernel2[i*iRow + (iRow-1-i)] = 1;

    } else {
        // 360-degree rotation
        std::cout << "WORTH NOTHING: 360-degree rotation \n";
        return;
    }
}

void tensorcv::imgprocKernel::upload_rotate(){

    if (repeat % 4 == 1) {
        cudaErrCheck( cudaMalloc((void **)&d_kernel1, iCol*iCol * sizeof(half)) );
        cudaErrCheck( cudaMemcpy(d_kernel1, kernel1, iCol*iCol  * sizeof(half), cudaMemcpyHostToDevice) );
        cudaErrCheck( cudaMalloc((void **)&d_temp1, iRow * iCol * sizeof(half)) );
        cudaErrCheck( cudaMalloc((void **)&d_temp2, iRow * iCol * sizeof(half)) );
        cudaErrCheck( cudaMalloc((void **)&d_temp3, iRow * iCol * sizeof(half)) );

    } else if (repeat % 4 == 2) {
        cudaErrCheck( cudaMalloc((void **)&d_kernel1, iCol*iCol * sizeof(half)) );
        cudaErrCheck( cudaMalloc((void **)&d_kernel2, iRow*iRow * sizeof(half)) );
        cudaErrCheck( cudaMemcpy(d_kernel1, kernel1, iCol*iCol  * sizeof(half), cudaMemcpyHostToDevice) );
        cudaErrCheck( cudaMemcpy(d_kernel2, kernel2, iRow*iRow  * sizeof(half), cudaMemcpyHostToDevice) );
        cudaErrCheck( cudaMalloc((void **)&d_temp1, iRow * iCol * sizeof(half)) );
        cudaErrCheck( cudaMalloc((void **)&d_temp2, iRow * iCol * sizeof(half)) );
        cudaErrCheck( cudaMalloc((void **)&d_temp3, iRow * iCol * sizeof(half)) );

    } else if (repeat % 4 == 3) {
        cudaErrCheck( cudaMalloc((void **)&d_kernel2, iRow*iRow * sizeof(half)) );
        cudaErrCheck( cudaMemcpy(d_kernel2, kernel2, iRow*iRow  * sizeof(half), cudaMemcpyHostToDevice) );
        cudaErrCheck( cudaMalloc((void **)&d_temp1, iRow * iCol * sizeof(half)) );
        cudaErrCheck( cudaMalloc((void **)&d_temp2, iRow * iCol * sizeof(half)) );
        cudaErrCheck( cudaMalloc((void **)&d_temp3, iRow * iCol * sizeof(half)) );

    } else {
        std::cout << "WORTH NOTHING: 360-degree rotation \n";
        return;
    }
}

void tensorcv::imgprocKernel::apply_rotate(cublasHandle_t handle, half* src1, half* src2, half* src3, half* dst1, half* dst2, half* dst3){
    const half alpha = 1.0;
    const half beta = 0.0;

    if (repeat % 4 == 1) {
        cublasErrCheck( cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, iRow, iCol, iCol,
                                        &alpha, src1, CUDA_R_16F, iCol, d_kernel1, CUDA_R_16F, iCol, 
                                        &beta, dst1, CUDA_R_16F, iRow,
                                        CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) );
        cublasErrCheck( cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, iRow, iCol, iCol,
                                        &alpha, src2, CUDA_R_16F, iCol, d_kernel1, CUDA_R_16F, iCol, 
                                        &beta, dst2, CUDA_R_16F, iRow,
                                        CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) );
        cublasErrCheck( cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, iRow, iCol, iCol,
                                        &alpha, src3, CUDA_R_16F, iCol, d_kernel1, CUDA_R_16F, iCol, 
                                        &beta, dst3, CUDA_R_16F, iRow,
                                        CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) );

    } else if (repeat % 4 == 2) {
        cublasErrCheck( cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, iRow, iCol, iCol,
                                        &alpha, src1, CUDA_R_16F, iCol, d_kernel1, CUDA_R_16F, iCol, 
                                        &beta, d_temp1, CUDA_R_16F, iRow,
                                        CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) );
        cublasErrCheck( cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, iRow, iCol, iCol,
                                        &alpha, src2, CUDA_R_16F, iCol, d_kernel1, CUDA_R_16F, iCol, 
                                        &beta, d_temp2, CUDA_R_16F, iRow,
                                        CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) );
        cublasErrCheck( cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, iRow, iCol, iCol,
                                        &alpha, src3, CUDA_R_16F, iCol, d_kernel1, CUDA_R_16F, iCol, 
                                        &beta, d_temp3, CUDA_R_16F, iRow,
                                        CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) );
        cublasErrCheck( cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, iRow, iCol, iRow,
                                        &alpha, d_kernel2, CUDA_R_16F, iRow, d_temp1, CUDA_R_16F, iRow, 
                                        &beta, dst1, CUDA_R_16F, iRow,
                                        CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) );
        cublasErrCheck( cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, iRow, iCol, iRow,
                                        &alpha, d_kernel2, CUDA_R_16F, iRow, d_temp2, CUDA_R_16F, iRow, 
                                        &beta, dst2, CUDA_R_16F, iRow,
                                        CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) );
        cublasErrCheck( cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, iRow, iCol, iRow,
                                        &alpha, d_kernel2, CUDA_R_16F, iRow, d_temp3, CUDA_R_16F, iRow, 
                                        &beta, dst3, CUDA_R_16F, iRow,
                                        CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) );

    } else if (repeat % 4 == 3) {
        cublasErrCheck( cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, iRow, iCol, iRow,
                                        &alpha, d_kernel2, CUDA_R_16F, iRow, src1, CUDA_R_16F, iCol, 
                                        &beta, dst1, CUDA_R_16F, iRow,
                                        CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) );
        cublasErrCheck( cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, iRow, iCol, iRow,
                                        &alpha, d_kernel2, CUDA_R_16F, iRow, src2, CUDA_R_16F, iCol, 
                                        &beta, dst2, CUDA_R_16F, iRow,
                                        CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) );
        cublasErrCheck( cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, iRow, iCol, iRow,
                                        &alpha, d_kernel2, CUDA_R_16F, iRow, src3, CUDA_R_16F, iCol, 
                                        &beta, dst3, CUDA_R_16F, iRow,
                                        CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) );

    } else {
        std::cout << "WORTH NOTHING: 360-degree rotation \n";
        return;
    }
}

void tensorcv::imgprocKernel::release_rotate(){
    if (repeat % 4 == 1 || repeat % 4 == 2) {
        cudaErrCheck( cudaFree(d_kernel1) );
        delete[] kernel1;
    }
    if (repeat % 4 == 2 || repeat % 4 == 3) {
        cudaErrCheck( cudaFree(d_kernel2) );
        delete[] kernel2;
    }
    cudaErrCheck( cudaFree(d_temp1) );
    cudaErrCheck( cudaFree(d_temp2) );
    cudaErrCheck( cudaFree(d_temp3) );
}

// // ****************************************************************************************************
// // Normalize
// // ****************************************************************************************************

void tensorcv::imgprocKernel::init_normalize(int iRow_, int iCol_, int channelCode_){
    iRow = iRow_; iCol = iCol_;
    channelCode = channelCode_;

    kernel1 = new half[16*iRow]();
    kernel2 = new half[iCol*16]();

    for (int i=0; i<iRow; i++) 
        kernel1[i] = (float)1/iRow;
    for (int j=0; j<iCol; j++) 
        kernel2[j*16] = (float)1/iCol;
}

void tensorcv::imgprocKernel::upload_normalize(){
    cudaErrCheck( cudaMalloc((void **)&d_kernel1, 16*iRow * sizeof(half)) );
    cudaErrCheck( cudaMalloc((void **)&d_kernel2, iCol*16 * sizeof(half)) );
    cudaErrCheck( cudaMemcpy(d_kernel1, kernel1, 16*iRow * sizeof(half), cudaMemcpyHostToDevice) );
    cudaErrCheck( cudaMemcpy(d_kernel2, kernel2, iCol*16 * sizeof(half), cudaMemcpyHostToDevice) );

    cudaErrCheck( cudaMalloc((void **)&d_temp1, 16 * 16 * sizeof(half)) );
    cudaErrCheck( cudaMalloc((void **)&d_temp2, (iRow+1) * 16 * sizeof(half)) );
    cudaErrCheck( cudaMalloc((void **)&d_temp3, (iRow+1) * iCol * sizeof(half)) );
}

__global__ void comptue_norm( half* dst, half* sum, half* squaredSum, half* input, int oRow, int oCol) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < oRow * oCol) {
        half mean = sum[0];
        half stddev = sqrt((float)squaredSum[0] - (float)mean * (float)mean);
        dst[idx] = ((float)input[idx] - (float)mean) / (float)stddev;
    }
}

void tensorcv::imgprocKernel::apply_normalize(cublasHandle_t handle, half* src1, half* src2, half* src3, half* dst1, half* dst2, half* dst3){
    const half alpha = 1.0;
    const half beta = 0.0;
    const half gamma = 1.0/iCol;

    // R
    cublasErrCheck( cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, 16, iCol, iRow,
                                    &alpha, d_kernel1, CUDA_R_16F, iRow, src1, CUDA_R_16F, iCol, 
                                    &beta, d_temp3, CUDA_R_16F, 16,
                                    CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) );
    cublasErrCheck( cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, 16, 16, iCol,
                                    &alpha, d_temp3, CUDA_R_16F, 16, d_kernel2, CUDA_R_16F, 16, 
                                    &beta, d_temp1, CUDA_R_16F, 16,
                                    CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) );

    cublasErrCheck( cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, iRow, iRow, iCol,
                                    &alpha, src1, CUDA_R_16F, iCol, src1, CUDA_R_16F, iCol, 
                                    &beta, d_temp3, CUDA_R_16F, iRow,
                                    CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) );
    cublasErrCheck( cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, iRow+1, 16, iRow,
                                    &gamma, d_temp3, CUDA_R_16F, iRow+1, d_kernel1, CUDA_R_16F, iRow, 
                                    &beta, d_temp2, CUDA_R_16F, iRow+1,
                                    CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) );

    comptue_norm<<<(iRow*iCol +31)/32, 32>>>(dst1, d_temp1, d_temp2, src1, iRow, iCol);

    // G
    cublasErrCheck( cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, 16, iCol, iRow,
                                    &alpha, d_kernel1, CUDA_R_16F, iRow, src2, CUDA_R_16F, iCol, 
                                    &beta, d_temp3, CUDA_R_16F, 16,
                                    CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) );
    cublasErrCheck( cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, 16, 16, iCol,
                                    &alpha, d_temp3, CUDA_R_16F, 16, d_kernel2, CUDA_R_16F, 16, 
                                    &beta, d_temp1, CUDA_R_16F, 16,
                                    CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) );

    cublasErrCheck( cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, iRow, iRow, iCol,
                                    &alpha, src2, CUDA_R_16F, iCol, src2, CUDA_R_16F, iCol, 
                                    &beta, d_temp3, CUDA_R_16F, iRow,
                                    CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) );
    cublasErrCheck( cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, iRow+1, 16, iRow,
                                    &gamma, d_temp3, CUDA_R_16F, iRow+1, d_kernel1, CUDA_R_16F, iRow, 
                                    &beta, d_temp2, CUDA_R_16F, iRow+1,
                                    CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) );
    
    comptue_norm<<<(iRow*iCol +31)/32, 32>>>(dst2, d_temp1, d_temp2, src2, iRow, iCol);

    // B
    cublasErrCheck( cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, 16, iCol, iRow,
                                    &alpha, d_kernel1, CUDA_R_16F, iRow, src3, CUDA_R_16F, iCol, 
                                    &beta, d_temp3, CUDA_R_16F, 16,
                                    CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) );
    cublasErrCheck( cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, 16, 16, iCol,
                                    &alpha, d_temp3, CUDA_R_16F, 16, d_kernel2, CUDA_R_16F, 16, 
                                    &beta, d_temp1, CUDA_R_16F, 16,
                                    CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) );

    cublasErrCheck( cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, iRow, iRow, iCol,
                                    &alpha, src3, CUDA_R_16F, iCol, src3, CUDA_R_16F, iCol, 
                                    &beta, d_temp3, CUDA_R_16F, iRow,
                                    CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) );
    cublasErrCheck( cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, iRow+1, 16, iRow,
                                    &gamma, d_temp3, CUDA_R_16F, iRow+1, d_kernel1, CUDA_R_16F, iRow, 
                                    &beta, d_temp2, CUDA_R_16F, iRow+1,
                                    CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) );

    comptue_norm<<<(iRow*iCol +31)/32,32>>>(dst3, d_temp1, d_temp2, src3, iRow, iCol);
}

void tensorcv::imgprocKernel::release_normalize(){
    cudaErrCheck( cudaFree(d_kernel1) );
    cudaErrCheck( cudaFree(d_kernel2) );
    cudaErrCheck( cudaFree(d_temp1) );
    cudaErrCheck( cudaFree(d_temp2) );
    cudaErrCheck( cudaFree(d_temp3) );
    delete[] kernel1;
    delete[] kernel2;
}

// ****************************************************************************************************
// Integrated
// ****************************************************************************************************

// GPU matmul function
__global__ void matmul(half* dst, half* src1, half* src2, int M, int N, int K, int transpose=0){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row >= M || col >= N){return;}

    half sum = __float2half(0.0);
    for(int i=0; i<K; i++)
        sum = __hadd(sum, __hmul(src1[row*K+i], src2[i*N+col]));
        // sum = (float)sum + (float)src1[row*K+i] * (float)src2[i*N+col];

    if (transpose == 1)
        dst[col*M+row] = sum;
    else
        dst[row*N+col] = sum;

}

void tensorcv::imgprocKernel::init_integrated(tensorcv::imgprocKernel kernel_rs, 
                                              tensorcv::imgprocKernel kernel_cp, 
                                              tensorcv::imgprocKernel kernel_cvt, 
                                              tensorcv::imgprocKernel kernel_rt){

    // kernel_resize -> Rs1 * input * Rs2
    // kernel_crop -> Cp1 * input * Cp2
    // kernel_cvt -> input * Cvt1
    // kernel_rt -> input * Rt1

    // set input and output size
    iRow = kernel_rs.iRow;
    iCol = kernel_rs.iCol;
    oRow = kernel_cp.oRow;
    oCol = kernel_cp.oCol;
    colorCode = kernel_cvt.colorCode;
    repeat = kernel_rt.repeat % 4;

    // T( Cp1 * Rs1 * In * Rs2 * Cp2 * Cvt1 * Rt1 )
    // T( Rs2 * Cp2 * Cv1 * Rt1 ) * T(Cp1 * Rs1 * In)
    // T(Rs2*Cp2) * T(Cv1 * Rt1) * T(Cp1 * Rs1 * In)
    // TT((T(Rs2*Cp2) * T(Cv1 * Rt1)) * T(Cp1 * Rs1 * In))

    kernel1 = new half[oRow*iRow]();
    kernel2 = new half[3*iCol*3*oCol]();
    int rowCropFactor = kernel_rs.oRow/2 - oRow/2;
    int colCropFactor = kernel_rs.oCol/2 - oCol/2;

    // Cp1 * Rs1
    for (int i=0; i<oRow; i++) {
        for (int j=0; j<iRow; j++) {
            kernel1[i*iRow + j] = kernel_rs.kernel1[(i+rowCropFactor)*iRow + j];
        }
    }
    // T(Rs2 * Cp2)
    for (int i=0; i<3*oCol; i++) {
        for (int j=0; j<3*iCol; j++) {
            kernel2[i*3*iCol + j] = kernel_rs.kernel2[(i+3*colCropFactor)*3*iCol + j];
        }
    }

    // T(Cvt1 * Rt1)
    kernel3 = new half[3*oCol*3*oCol]();
    if (repeat != 1) {std::cout << "Not implemented \n"; return;}

    for (int i=0; i<oCol; i++) {
        kernel3[(3*i)*3*oCol + 3*i+2] = colorPallet[colorCode][0];
        kernel3[(3*i)*3*oCol + 3*i+1] = colorPallet[colorCode][1];
        kernel3[(3*i)*3*oCol + 3*i] = colorPallet[colorCode][2];
        kernel3[(3*i+1)*3*oCol + 3*i+2] = colorPallet[colorCode][3];
        kernel3[(3*i+1)*3*oCol + 3*i+1] = colorPallet[colorCode][4];
        kernel3[(3*i+1)*3*oCol + 3*i] = colorPallet[colorCode][5];
        kernel3[(3*i+2)*3*oCol + 3*i+2] = colorPallet[colorCode][6];
        kernel3[(3*i+2)*3*oCol + 3*i+1] = colorPallet[colorCode][7];
        kernel3[(3*i+2)*3*oCol + 3*i] = colorPallet[colorCode][8];
    }

    kernel4 = new half[3*oCol*3*oCol]();
    for (int i=0; i<3*oCol; i++)
        kernel4[i*3*oCol + (3*oCol-1-i)] = 1;

    cudaMalloc((void**)&d_kernel3, 3*oCol * 3*oCol * sizeof(half));
    cudaMemcpy(d_kernel3, kernel3, 3*oCol * 3*oCol * sizeof(half), cudaMemcpyHostToDevice);
    
    cudaMalloc((void**)&d_kernel4, 3*oCol * 3*oCol * sizeof(half));
    cudaMemcpy(d_kernel4, kernel4, 3*oCol * 3*oCol * sizeof(half), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_kernel5, 3*oCol * 3*oCol * sizeof(half));
    cudaMemset(d_kernel5, 0, 3*oCol * 3*oCol * sizeof(half));

    matmul<<<dim3((3*oCol+63)/64,3*oCol), dim3(64)>>>(d_kernel5, d_kernel3, d_kernel4, 3*oCol, 3*oCol, 3*oCol, 1);
    
    // load and print d_kernel3
    // half* h_kernel5 = new half[3*oCol*3*oCol]();
    // cudaMemcpy(h_kernel5, d_kernel5, 3*oCol * 3*oCol * sizeof(half), cudaMemcpyDeviceToHost);
    // for (int i=0; i<3*oCol; i++){
    //     for (int j=0; j<3*oCol; j++){
    //         std::cout << (float) h_kernel5[i*3*oCol + j] << " ";
    //     }
    //     std::cout << "\n";
    // }

    // T(Cvt1 * Rt1) * T(Rs2 * Cp2)
    cudaMalloc((void**)&d_kernel2, 3*iCol * 3*oCol * sizeof(half));
    cudaMemcpy(d_kernel2, kernel2, 3*iCol * 3*oCol * sizeof(half), cudaMemcpyHostToDevice);

    cudaFree(d_kernel3);
    cudaMalloc((void**)&d_kernel3, 3*iCol * 3*oCol * sizeof(half));
    cudaMemset(d_kernel3, 0, 3*iCol * 3*oCol * sizeof(half));

    matmul<<<dim3((3*iCol+63)/64,3*oCol), dim3(64)>>>(d_kernel3, d_kernel5, d_kernel2, 3*oCol, 3*iCol, 3*oCol);
    // cudaMemcpy(kernel2, d_kernel3, 3*iCol * 3*oCol * sizeof(half), cudaMemcpyDeviceToHost);

    delete[] kernel3;
    delete[] kernel4;
    cudaFree(d_kernel2);
    cudaFree(d_kernel4);
    cudaFree(d_kernel5);
}

void tensorcv::imgprocKernel::upload_integrated(){
    cudaErrCheck( cudaMalloc((void **)&d_kernel1, oRow * iRow * sizeof(half)) );
    cudaErrCheck( cudaMemcpy(d_kernel1, kernel1, oRow * iRow  * sizeof(half), cudaMemcpyHostToDevice) );
    // cudaErrCheck( cudaMalloc((void **)&d_kernel2, iCol*3 * oCol*3 * sizeof(half)) );
    // cudaErrCheck( cudaMemcpy(d_kernel2, kernel2, iCol*3 * oCol*3  * sizeof(half), cudaMemcpyHostToDevice) );
    cudaErrCheck( cudaMalloc((void **)&d_temp1, oRow * 3 * iCol * sizeof(half)) );
}

void tensorcv::imgprocKernel::apply_integrated(cublasHandle_t handle, half* src, half* dst){
    const half alpha = 1.0;
    const half beta = 0.0;
    // T(Cp1 * Rs1 * In)
    cublasErrCheck( cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, oRow, 3*iCol, iRow,
                                    &alpha, d_kernel1, CUDA_R_16F, iRow, src, CUDA_R_16F, 3*iCol, 
                                    &beta, d_temp1, CUDA_R_16F, oRow,
                                    CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) );
    // T( T(Cvt1 * Rt1) * T(Rs2 * Cp2) * T(Cp1 * Rs1 * In))
    cublasErrCheck( cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, 3*oCol, oRow, 3*iCol, 
                                    &alpha, d_kernel3, CUDA_R_16F, 3*iCol, d_temp1, CUDA_R_16F, oRow, 
                                    &beta, dst, CUDA_R_16F, 3*oCol,
                                    CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) );
}

void tensorcv::imgprocKernel::release_integrated(){
    cudaFree(d_kernel1);
    cudaFree(d_kernel3);
    cudaFree(d_temp1);
}