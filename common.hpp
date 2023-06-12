#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>

#include <math.h>
#include <chrono>

namespace tensorcv {
enum COLORCODE {
    RGB2BGR,
    BGR2RGB,
    RGB2YUV,
    YUV2RGB,
    BGR2YUV,
    YUV2BGR
};
}