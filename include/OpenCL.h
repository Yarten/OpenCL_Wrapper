//
// Created by yarten on 18-4-10.
//

#pragma once
#include <CL/cl.hpp>
#include <string>
#include <vector>
#include <map>
#include <initializer_list>
#include <list>
#include "Buffer.h"
#include "Event.h"
#include <iostream>

using namespace std;

namespace opencl
{
    class OpenCL
    {
    public:

        class Program : public cl::Program
        {
        public:

            class Kernel : public cl::Kernel
            {
            public:

                Kernel() = default;

                void SetSize(size_t local);

                void SetSize(size_t global, size_t local);

                void SetSize(size_t ndRange, size_t global, size_t local);

                template <typename ...Args>
                Event operator()(Buffer<Args> & ... params)
                {
                    cl::Event event;
                    cl_int status;
                    buffers.clear();
                    SetArg(0, params...);
                    status = GetCommandQueue(platformIndex, deviceIndex).enqueueNDRangeKernel(*this, cl::NullRange, cl::NDRange(global, local), cl::NullRange, nullptr, &event);
                    PrintError(status, "Enqueue NDRange Kernel");
                    status = GetCommandQueue(platformIndex, deviceIndex).finish();
                    PrintError(status, "Running Kernel");
                    CopyBack(0, params...);
                    return Event(event);
                };

                Event operator()();

            private:

                template <typename T>
                void SetArg(cl_uint i, Buffer<T> & buffer)
                {
                    if(i >= flags.size())
                        throw "OpenCL Function Call's Parameters doesn't Match Declaration.";

                    cl_mem_flags flag = (buffer.Flag() == 0 ? flags[i] : buffer.Flag());
                    cl_int status;
                    if(flag == 0)
                    {
                        status = Kernel::setArg(i, buffer);
                    }
                    else
                    {
                        buffers.emplace_back(GetContext(platformIndex, deviceIndex), flag, buffer.Size(), buffer.Data());
                        status = Kernel::setArg(i, buffers.back());
                    }

                    PrintError(status, "Setting Parameter");
                }

                template <typename T, typename ... Args>
                void SetArg(cl_uint i, Buffer<T> & buffer, Buffer<Args> & ... rests)
                {
                    SetArg(i, buffer);
                    SetArg(i+1, rests...);
                };

                template <typename T>
                void CopyBack(cl_uint i, Buffer<T> & buffer)
                {
                    cl_mem_flags flag = (buffer.Flag() == 0 ? flags[i] : buffer.Flag());
                    if((flag & CL_MEM_WRITE_ONLY) || (flag & CL_MEM_READ_WRITE))
                    {
                        T * ptr = GetCommandQueue(platformIndex, deviceIndex).enqueueMapBuffer(
                                buffers.front(),
                                CL_TRUE,
                                CL_MAP_READ,
                                0,
                                buffer.Size());
                        buffers.pop_front();

                        if((flag & CL_MEM_USE_HOST_PTR) == 0)
                        {
                            buffer.Set(ptr, buffer.Size());
                        }
                    }

                }

                template <typename T, typename ... Args>
                void CopyBack(cl_uint i, Buffer<T> & buffer, Buffer<Args> & ... rests)
                {
                    CopyBack(i, buffer);
                    CopyBack(i+1, rests...);
                };

                void PrintError(cl_int status, const string & msg)
                {
                    if(status != CL_SUCCESS)
                    {
                        cerr << "Error " << msg << " [code " << status << "]" << endl;
                    }
                }

            private:

                friend class OpenCL::Program;

                Kernel(Program & program, int platformIndex, int deviceIndex, const string & name, const string & head);

                vector<cl_mem_flags> flags;
                int platformIndex, deviceIndex;
                size_t ndRange, global, local;
                list<cl::Buffer> buffers;
            };

            Kernel & operator()(const string & name);

            Program() = default;

        private:
            friend class OpenCL;
            friend class Kernel;

            Program(int platformIndex, int deviceIndex, const string & content, const string & buildOptions = "");

            map<string, Kernel> kernels;
            int platformIndex, deviceIndex;
        };

        friend class Program;
        friend class Kernel;

    public:

        OpenCL(int platformIndex = 0, int deviceIndex = 0);

        Program & operator()(const string & program); // 不直接使用重载是因为在clion下代码提示不正常

        Program & operator()(const string & program, const string & buildOptions);

        Program & Load(const string & program, const string & buildOptions = "");

    public: /// OpenCL平台信息获取接口

        static size_t GetPlatformsCount() { return platforms.size(); }
        static size_t GetDevicesCount(size_t platformIndex = -1);
        static string GetPlatformsInformation();
        static string GetPlatformInformation(size_t index);
        static string GetDevicesInformation(size_t platformIndex = -1);
        static string GetDeviceInformation(size_t platformIndex, size_t deviceIndex);

    private: /// OpenCL平台管理函数

        static int CreateOpenCLEnvironment();
        static int GlobalInitializationHelper;
        static cl::Context & GetContext(int i, int j) { return contexts[i][j]; }
        static cl::CommandQueue & GetCommandQueue(int i, int j) { return queues[i][j]; }
        static cl::Device & GetDevice(int i, int j) { return devices[i][j]; }

    private: /// OpenCL平台管理成员

        static vector<cl::Platform> platforms;
        static vector<vector<cl::Device>> devices;
        static vector<vector<cl::Context>> contexts;
        static vector<vector<cl::CommandQueue>> queues;

    private: /// OpenCL某个设备的成员

        cl::Context & context;
        cl::CommandQueue & queue;
        cl::Device & device;
        const int platformIndex, deviceIndex;
        map<string, Program> programs;
    };
}