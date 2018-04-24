//
// Created by yarten on 18-4-24.
//

#include <Event.h>
using namespace opencl;

Event::Event(const cl::Event &event)
    : cl::Event(event),
      ExecutionTime((event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>())* 1e-6f)
{

}
