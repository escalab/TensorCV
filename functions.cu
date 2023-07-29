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
    if (transpose)
        dst[ty*cols + tx] = (unsigned char)((float)src[tx*rows + ty] * alpha);
    else
        dst[ty*cols + tx] = (unsigned char)((float)src[ty*cols + tx] * alpha);
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
    
    dim3 block(16, 16);
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

void tensorcv::imgprocKernel::init_cvtcolor(int iRow_, int iCol_, int colorCode){
    iRow = iRow_; iCol = iCol_;

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
__global__ void matmul(half* src1, half* src2, half* dst, int M, int N, int K, int transpose=0){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < M && col < N){
        half sum = __float2half(0.0);
        for(int i=0; i<K; i++){
            sum = (float)sum + (float)src1[row*K+i] * (float)src2[i*N+col];
        }
        if (transpose == 1){
            dst[col*M+row] = sum;
        }
        else{
            dst[row*N+col] = sum;
        }
    }
}

void tensorcv::imgprocKernel::init_integrated(int iRow_, int iCol_, int rRow, int rCol, int cRow, int cCol, int colorCode, int repeat_){
    
    iRow = iRow_;
    iCol = iCol_;
    oRow = cRow;
    oCol = cCol;
    repeat = repeat_ % 4;

    float rowResizeFactor = (float)rRow / (float)iRow;
    float colResizeFactor = (float)rCol / (float)iCol;
    float rowCropFactor = rRow/2 - cRow/2;
    float colCropFactor = rCol/2 - cCol/2;

    kernel1 = new half[cRow * iRow]();
    kernel2 = new half[3*iCol * 3*cCol]();

    for (int j=rowCropFactor; j<rRow-rowCropFactor; j++) {
        int i = j - rowCropFactor;
        int top = floor(j/rowResizeFactor);
        int bot = ceil(j/rowResizeFactor);
        float rowWeight = (float)j/rowResizeFactor - top;
        if (rowWeight == 0) {
            kernel1[i*iRow + top] = 1;
        } else {
            kernel1[i*iRow + top] = (float)(1 - rowWeight);
            kernel1[i*iRow + bot] = (float)(rowWeight);    
        }
    }

    for (int j=colCropFactor; j<rCol-colCropFactor; j++) { 
        int i = j - colCropFactor;   
        int left = floor(j/colResizeFactor);
        int right = ceil(j/colResizeFactor);
        float colWeight = (float)j/colResizeFactor - left;
        if (colWeight == 0) {
            kernel2[(3*i)*3*iCol + 3*left] = 1;
            kernel2[(3*i+1)*3*iCol + 3*left+1] = 1;
            kernel2[(3*i+2)*3*iCol + 3*left+2] = 1;
        } else {
            kernel2[(3*i)*3*iCol + 3*left] = (half)(1 - colWeight);
            kernel2[(3*i+1)*3*iCol + 3*left+1] = (half)(1 - colWeight);
            kernel2[(3*i+2)*3*iCol + 3*left+2] = (half)(1 - colWeight);
            kernel2[(3*i)*3*iCol + 3*right] = (half)(colWeight);
            kernel2[(3*i+1)*3*iCol + 3*right+1] = (half)(colWeight);
            kernel2[(3*i+2)*3*iCol + 3*right+2] = (half)(colWeight);
        }
    }

    kernel3 = new half[3*cCol*3*cCol]();
    bool skip_rotate = false;

    // for (int i=0; i<cCol; i++) {
    //     kernel3[(3*i)*3*cCol + 3*i] = colorPallet[colorCode][0];
    //     kernel3[(3*i)*3*cCol + 3*i+1] = colorPallet[colorCode][1];
    //     kernel3[(3*i)*3*cCol + 3*i+2] = colorPallet[colorCode][2];
    //     kernel3[(3*i+1)*3*cCol + 3*i] = colorPallet[colorCode][3];
    //     kernel3[(3*i+1)*3*cCol + 3*i+1] = colorPallet[colorCode][4];
    //     kernel3[(3*i+1)*3*cCol + 3*i+2] = colorPallet[colorCode][5];
    //     kernel3[(3*i+2)*3*cCol + 3*i] = colorPallet[colorCode][6];
    //     kernel3[(3*i+2)*3*cCol + 3*i+1] = colorPallet[colorCode][7];
    //     kernel3[(3*i+2)*3*cCol + 3*i+2] = colorPallet[colorCode][8];
    // }
    if (repeat == 1) {
        for (int i=0; i<cCol; i++) {
            kernel3[(3*i)*3*cCol + (i+2*cCol)] = colorPallet[colorCode][0];
            kernel3[(3*i)*3*cCol + (i+cCol)] = colorPallet[colorCode][1];
            kernel3[(3*i)*3*cCol + i] = colorPallet[colorCode][2];
            kernel3[(3*i+1)*3*cCol + (i+2*cCol)] = colorPallet[colorCode][3];
            kernel3[(3*i+1)*3*cCol + (i+cCol)] = colorPallet[colorCode][4];
            kernel3[(3*i+1)*3*cCol + i] = colorPallet[colorCode][5];
            kernel3[(3*i+2)*3*cCol + (i+2*cCol)] = colorPallet[colorCode][6];
            kernel3[(3*i+2)*3*cCol + (i+cCol)] = colorPallet[colorCode][7];
            kernel3[(3*i+2)*3*cCol + i] = colorPallet[colorCode][8];
        }
    } else if (repeat == 2) {
        ; // TODO
    } else if (repeat == 3) {
        ; // TODO
    } else {
        skip_rotate = true;
    }

    if (skip_rotate) {
        cudaMalloc((void**)&d_kernel1, cRow * iRow * sizeof(half));
        cudaMalloc((void**)&d_kernel2, 3*iCol * 3*cCol * sizeof(half));
        cudaMalloc((void**)&d_kernel3, 3*cCol * 3*cCol * sizeof(half));
        cudaMemcpy(d_kernel1, kernel1, cRow * iRow * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_kernel2, kernel2, 3*iCol * 3*cCol * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_kernel3, kernel3, 3*cCol * 3*cCol * sizeof(half), cudaMemcpyHostToDevice);

        matmul<<<dim3((3*iCol+31)/32,3*cCol), dim3(32,1)>>>(d_kernel2, d_kernel3, d_kernel2, 3*iCol, 3*cCol, 3*cCol);

        delete[] kernel3;
        cudaFree(d_kernel3);
    }

    if (repeat % 4 == 1) {
        kernel3 = new half[3*cCol*3*cCol]();
        for (int i=0; i<3*cCol; i++)
            kernel3[i*3*cCol + (3*cCol-1-i)] = 1;
        cudaMalloc((void**)&d_kernel3, 3*cCol * 3*cCol * sizeof(half));
        matmul<<<dim3((3*iCol+31)/32,3*cCol), dim3(32,1)>>>(d_kernel2, d_kernel3, d_kernel2, 3*iCol, 3*cCol, 3*cCol);
        delete[] kernel3;
        cudaFree(d_kernel3);

    } else if (repeat % 4 == 2) {
        kernel3 = new half[3*cCol*3*cCol]();
        kernel4 = new half[cRow*cRow]();
        for (int i=0; i<3*cCol; i++)
            kernel3[i*3*cCol + (3*cCol-1-i)] = 1;
        for (int i=0; i<cRow; i++)
            kernel4[i*cRow + (cRow-1-i)] = 1;
        cudaMalloc((void**)&d_kernel3, 3*cCol * 3*cCol * sizeof(half));
        cudaMalloc((void**)&d_kernel4, cRow * cRow * sizeof(half));
        matmul<<<dim3((cRow+31)/32,iRow), dim3(32,1)>>>(d_kernel4, d_kernel1, d_kernel1, cRow, iRow, cRow); 
        matmul<<<dim3((3*iCol+31)/32,3*cCol), dim3(32,1)>>>(d_kernel2, d_kernel3, d_kernel2, 3*iCol, 3*cCol, 3*cCol);
        delete[] kernel3;
        delete[] kernel4;
        cudaFree(d_kernel3);
        cudaFree(d_kernel4);
        
    } else if (repeat % 4 == 3) {
        kernel4 = new half[3*cRow*3*cRow]();
        for (int i=0; i<3*cRow; i++)
            kernel4[i*3*cRow + (3*cRow-1-i)] = 1;
        cudaMalloc((void**)&d_kernel4, cRow * cRow * sizeof(half));
        matmul<<<dim3((cRow+31)/32,iRow), dim3(32,1)>>>(d_kernel4, d_kernel1, d_kernel1, cRow, iRow, cRow); 
        delete[] kernel4;
        cudaFree(d_kernel4);

    } else {
        std::cout << "WORTH NOTHING: 360-degree rotation \n";
        return;
    }
}

void tensorcv::imgprocKernel::upload_integrated(){
    cudaErrCheck( cudaMalloc((void **)&d_kernel1, oRow * iRow * sizeof(half)) );
    cudaErrCheck( cudaMemcpy(d_kernel1, kernel1, oRow * iRow  * sizeof(half), cudaMemcpyHostToDevice) );
    cudaErrCheck( cudaMalloc((void **)&d_kernel2, iCol*3 * oCol*3 * sizeof(half)) );
    cudaErrCheck( cudaMemcpy(d_kernel2, kernel2, iCol*3 * oCol*3  * sizeof(half), cudaMemcpyHostToDevice) );
    cudaErrCheck( cudaMalloc((void **)&d_temp1, oRow * 3 * iCol * sizeof(half)) );

}

void tensorcv::imgprocKernel::apply_integrated(cublasHandle_t handle, half* src, half* dst){
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

void tensorcv::imgprocKernel::release_integrated(){
    cudaFree(d_kernel1);
    cudaFree(d_kernel2);
    cudaFree(d_temp1);
}

// ****************************************************************************************************
// GEMM Test
// ****************************************************************************************************

// void tensorcv::GEMMtest() {
//     cublasHandle_t handle;
//     cublasCreate(&handle);

//     int M = 4;
//     int N = 4;
//     int K = 4;

//     half *h_A, *h_B;
//     half *h_C;
//     cudaMallocHost(&h_A, M * K * sizeof(half));
//     cudaMallocHost(&h_B, K * N * sizeof(half));
//     cudaMallocHost(&h_C, M * N * sizeof(half));

//     // initialize h_A and h_B
//     for (int i = 0; i < M * K; i++)
//         h_A[i] = (half)(rand() % 128);
//     for (int i = 0; i < K * N; i++)
//         h_B[i] = (half)(rand() % 2);

//     // copy h_A and h_B to device
//     half *d_A, *d_B;
//     half *d_C;
//     cudaMalloc(&d_A, M * K * sizeof(half));
//     cudaMalloc(&d_B, K * N * sizeof(half));
//     cudaMalloc(&d_C, M * N * sizeof(half));
//     cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice);
//     cudaMemset(d_C, 0, M * N * sizeof(half));

//     for(int i=0; i<M; i++) {
//         for(int j=0; j<K; j++) {
//             std::cout << (float)h_A[i*K+j] << " ";
//         }
//         std::cout << std::endl;
//     }
//     std::cout << std::endl;
//     for(int i=0; i<K; i++) {
//         for(int j=0; j<N; j++) {
//             std::cout << (float)h_B[i*N+j] << " ";
//         }
//         std::cout << std::endl;
//     }
//     std::cout << std::endl;

//     // launch kernel
//     const half alpha = 1;
//     const half beta = 0;

//     long long execution_time = 0;
//     for (int i=0; i<100; i++) {
//         auto blockStart = std::chrono::high_resolution_clock::now();
//         cublasErrCheck( cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, 
//                                      &alpha, d_A, CUDA_R_16F, K, d_B, CUDA_R_16F, N, 
//                                      &beta, d_C, CUDA_R_16F, M, 
//                                      CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) );
//         // cublasErrCheck( cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, M/2, N, K, 
//         //                              &alpha, d_A, CUDA_R_16F, K*2, d_B, CUDA_R_16F, N, 
//         //                              &beta, d_C, CUDA_R_16F, M/2,
//         //                              CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) );
//         if(i != 0)
//             execution_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now()-blockStart).count();
//     }
//     std::cout << "Imgproc: " << execution_time/99 << std::endl;

//     cudaMemcpy(h_C, d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost);

//     // C is transposed
//     for(int i=0; i<M; i++) {
//         for(int j=0; j<N; j++) {
//             std::cout << (float)h_C[j*M+i] << " ";
//         }
//         std::cout << std::endl;
//     }
//     std::cout << std::endl;

//     // for(int i=0; i<M; i++) {
//     //     for(int j=0; j<N; j++) {
//     //         std::cout << (float)h_C[i*N+j] << " ";
//     //     }
//     //     std::cout << std::endl;
//     // }
//     // std::cout << std::endl;

//     // Free the memory
//     cudaFree(d_A);
//     cudaFree(d_B);
//     cudaFree(d_C);
// }