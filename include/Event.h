//
// Created by yarten on 18-4-24.
//

#pragma once

#include <CL/cl.hpp>

namespace opencl
{
    class Event : public cl::Event
    {
    public:

        Event(const cl::Event & event);

        const float ExecutionTime;
    };
}