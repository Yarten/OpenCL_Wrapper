//
// Created by yarten on 18-4-10.
//

#include "../include/OpenCL.h"
#include <CL/cl.h>
#include <CL/cl.hpp>
#include <functional>
#include <thread>
#include <fstream>
#include <sstream>
#include <iostream>
#include <boost/algorithm/string.hpp>
#include <regex>

using namespace cl;
using namespace boost;
using namespace opencl;
using std::size_t;

vector<cl::Platform> OpenCL::platforms;
vector<vector<cl::Device>> OpenCL::devices;
vector<vector<cl::Context>> OpenCL::contexts;
vector<vector<cl::CommandQueue>> OpenCL::queues;
int OpenCL::GlobalInitializationHelper = OpenCL::CreateOpenCLEnvironment();

OpenCL::Program &OpenCL::Load(const string &program, const string & buildOptions)
{
    ifstream file(program);
    stringstream stream;
    stream << file.rdbuf();
    string content(stream.str());

    programs[program] = Program(platformIndex, deviceIndex, content, buildOptions);
    return programs[program];
}

int OpenCL::CreateOpenCLEnvironment()
{
    cl_int status;

    /* Get Platform and Device Info */
    status = Platform::get(&platforms);
    std::size_t N = platforms.size();

    devices.resize(N);
    contexts.resize(N);
    queues.resize(N);


    for(std::size_t i = 0, size = platforms.size(); i < size; i++)
    {
        platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &devices[i]);

        for(std::size_t j = 0, size2 = devices[i].size(); j < size2; j++)
        {
            contexts[i].emplace_back(devices[i][j]);
            queues[i].emplace_back(contexts[i][j], devices[i][j], CL_QUEUE_PROFILING_ENABLE);
        }
    }

    return status;
}


string OpenCL::GetPlatformInformation(std::size_t index)
{
    if(index >= platforms.size())
    {
        stringstream stream;
        stream << "Wrong Index: " << index << " out of range [" << 0 << ", " << platforms.size()-1 << "]";
        return stream.str();
    }

    return platforms[index].getInfo<CL_PLATFORM_NAME>();
}

string OpenCL::GetPlatformsInformation()
{
    stringstream stream;
    stream << "++++++++++++++++++++++++++++++++++++++" << endl;
    for(std::size_t i = 0, size = platforms.size(); i < size; i++)
    {
        stream << GetPlatformInformation(i) << endl;
        stream << "++++++++++++++++++++++++++++++++++++++" << endl;
    }
    return stream.str();
}

std::size_t OpenCL::GetDevicesCount(std::size_t platformIndex)
{
    if(platformIndex >= platforms.size())
    {
        std::size_t sum = 0;
        for(auto & i : devices)
            sum += i.size();
        return sum;
    }

    return devices[platformIndex].size();
}

string OpenCL::GetDevicesInformation(std::size_t index)
{
    stringstream stream;
    stream << "++++++++++++++++++++++++++++++++++++++" << endl;

    if(index < 0 || index >= devices.size())
    {
        for(std::size_t i = 0, size = devices.size(); i < size; i++)
        {
            for(std::size_t j = 0, size2 = devices[i].size(); j < size2; j++)
            {
                stream << "Platform " << i << " Device " << j << endl;
                stream << GetDeviceInformation(i, j);
                stream << "++++++++++++++++++++++++++++++++++++++" << endl;
            }
        }
    }
    else
    {
        for(std::size_t i = 0, size = devices[index].size(); i < size; i++)
        {
            stream << GetDeviceInformation(index, i) << endl;
            stream << "++++++++++++++++++++++++++++++++++++++" << endl;
        }
    }

    return stream.str();
}

string OpenCL::GetDeviceInformation(std::size_t platformIndex, std::size_t deviceIndex)
{
    stringstream stream;
    stream << devices[platformIndex][deviceIndex].getInfo<CL_DEVICE_NAME>() << "\n";
    stream << "Address Bits: " << devices[platformIndex][deviceIndex].getInfo<CL_DEVICE_ADDRESS_BITS>() << "\n";
    stream << "Support: " << devices[platformIndex][deviceIndex].getInfo<CL_DEVICE_EXTENSIONS>() << "\n";
    return stream.str();
}


OpenCL::OpenCL(int i, int j)
    : context(contexts[i][j]), queue(queues[i][j]), device(devices[i][j]), platformIndex(i), deviceIndex(j)
{
}

OpenCL::Program &OpenCL::operator()(const string &program)
{
    return operator()(program, "");
}


OpenCL::Program &OpenCL::operator()(const string &program, const string & buildOptions)
{
    if(programs.find(program) == programs.end())
        return Load(program, buildOptions);
    else
        return programs[program];
}


OpenCL::Program::Program(int platformIndex, int deviceIndex, const string &content, const string & buildOptions)
    : cl::Program(GetContext(platformIndex, deviceIndex), content, false), platformIndex(platformIndex), deviceIndex(deviceIndex)
{
    //region 编译cl文件，报错则退出
    cl_int status = cl::Program::build(buildOptions.c_str());
    if(status != CL_SUCCESS)
    {
        cerr << "Build Program Fail" << endl;
        cerr << cl::Program::getBuildInfo<CL_PROGRAM_BUILD_LOG>(GetDevice(platformIndex, deviceIndex)) << endl;
        return;
    }
    else
        cout << "Build Program Successfully" << endl;
    //endregion

    //region 分割出每个kernel的函数签名
    string s = cl::Program::getInfo<CL_PROGRAM_KERNEL_NAMES>();
    vector<string> kernelNames;
    vector<string> heads;

    split(kernelNames, s, boost::is_any_of(";"));

    std::smatch sm;
    s = content;
    std::regex expr(R"(__kernel\s*void*\s*[\w]*\s*\(((?!\{)([\s\S]))*\))");
    while(std::regex_search(s, sm, expr))
    {
        heads.push_back(sm[0]);
        s = sm.suffix().str();
    }
    //endregion

    //region 创建kernel
    for(std::size_t i = 0, size = kernelNames.size(); i < size; i++)
    {
        kernels[kernelNames[i].c_str()] = Kernel(*this, platformIndex, deviceIndex, kernelNames[i], heads[i]);
    }
    //endregion
}


OpenCL::Program::Kernel & OpenCL::Program::operator()(const string &name)
{
    if(kernels.find(name) == kernels.end())
        return kernels[""];
    else
        return kernels[name];
}

OpenCL::Program::Kernel::Kernel(Program & program, int platformIndex, int deviceIndex, const string &name, const string &head)
    : cl::Kernel(program, name.c_str()), platformIndex(platformIndex), deviceIndex(deviceIndex)
{
    //region 分割出参数部分
    std::smatch sm;
    std::regex expr(R"(\([\s\S]*\))");
    std::regex_search(head, sm, expr);
    string s = sm[0];
    boost::trim(s);
    s = s.substr(1, s.length()-2);
    //endregion

    //region 分离每一个参数，并保存其读写性、来源性、被读写性等标识
    vector<string> tokens;
    boost::split(tokens, s, boost::is_any_of(","));
    std::size_t size = tokens.size();
    flags.resize(size);

    for(std::size_t i = 0; i < size; i++)
    {
        if(tokens[i].find("*") != string::npos)
        {
            if(tokens[i].find("const") != string::npos)
                flags[i] = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;
            else if(tokens[i].find("read_only") != string::npos)
                flags[i] = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;
            else if(tokens[i].find("write_only") != string::npos)
                flags[i] = CL_MEM_WRITE_ONLY;
            else
                flags[i] = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;
        }
        else
            flags[i] = 0;
    }
    //endregion
}

void OpenCL::Program::Kernel::SetSize(cl::NDRange global, cl::NDRange local)
{
    this->global = global;
    this->local = local;
}

opencl::Event OpenCL::Program::Kernel::operator()()
{
    cl::Event event;
    cl_int status;
    status = GetCommandQueue(platformIndex, deviceIndex).enqueueNDRangeKernel(*this, cl::NullRange, global, local, nullptr, &event);
    PrintError(status, "Enqueue NDRange Kernel");
    status = GetCommandQueue(platformIndex, deviceIndex).finish();
    PrintError(status, "Running Kernel");
    return Event(event);
}