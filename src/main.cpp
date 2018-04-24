//
// Created by yarten on 18-4-17.
//

#include "OpenCL.h"
#include <iostream>
#include "Buffer.h"
#include <array>
#include <initializer_list>

using namespace std;
using namespace opencl;

int main()
{
    cout << OpenCL::GetDevicesInformation() << endl;

    OpenCL openCL;
    OpenCL::Program::Kernel kernel = openCL("./Vadd.cl")("vadd");

    size_t rows = 200, cols = 300;
    kernel.SetSize({rows, cols});
    Buffer<float> a(rows * cols), b(rows * cols), c(rows * cols);

    float d[200 * 300];
    Buffer<float> e(d, 200 * 300, true);

    for(size_t i = 0; i < rows; i++)
        for(size_t j = 0; j < cols; j++)
        {
            a[i * cols + j] = 1;
            b[i * cols + j] = 2;
            d[i * cols + j] = 3;
        }

    Event event = kernel(c, a, e);

    cout << endl << event.ExecutionTime << endl;
    cout << c[0] << endl;

    return 0;
}