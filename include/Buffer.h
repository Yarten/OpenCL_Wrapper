//
// Created by yarten on 18-4-23.
//

#pragma once

#include <cstdlib>
#include <CL/cl.hpp>
#include <memory>
#include <exception>
#include <sstream>

using std::shared_ptr;
using std::runtime_error;
using std::stringstream;

namespace opencl
{
    template <typename T>
    class Buffer
    {
    public:
        /// 构造器
        explicit Buffer(size_t size = -1, cl_mem_flags flag = 0, T defaultValue = T())
                : size(size), flag(flag)
        {
            if(size == -1)
            {
                data.reset(new T(defaultValue));
                this->flag = 0;
            }
            else
                Set(size, flag, defaultValue);
        }

        Buffer(shared_ptr<T> data, size_t size, cl_mem_flags flag = 0)
        {
            Set(data, size, flag);
        }

        Buffer(const T * data, size_t size, cl_mem_flags flag = 0)
        {
            Set(data, size, flag);
        }

        /// 赋值接口
        void Set(size_t size, cl_mem_flags flag = -1, T defaultValue = T())
        {
            this->size = size;
            if(flag != -1)
                this->flag = flag;
            data.reset(new T[size]);
            for(int i = 0; i < size; i++)
                data.get()[i] = defaultValue;
        }

        void Set(shared_ptr<T> data, size_t size, cl_mem_flags flag = 0)
        {
            this->size = size;
            this->flag = flag;
            this->data = data;
        }

        void Set(const T * data, size_t size, cl_mem_flags flag = 0)
        {
            this->size = size;
            this->flag = flag;
            this->data.reset(new T[size]);
            for(int i = 0; i < size; i++)
                this->data.get()[i] = data[i];
        }

        /// 信息获取接口
        T * Data() { return data.get(); }

        size_t Size() { return size * sizeof(T); }

        size_t Length() { return size; }

        cl_mem_flags Flag() { return flag; }

        /// 值操作接口
        Buffer & operator=(T value)
        {
            if(size != -1)
            {
                size = -1;
                this->flag = 0;
                data.reset(new T(value));
            }
            else
                *data = value;

            return *this;
        }

        operator T() const { return *data; }

        operator T&() { return *data; }

        /// 数组操作接口
        T & operator[](size_t index)
        {
            Check(index, true);
            return data.get()[index];
        }

        const T & operator[](size_t index) const
        {
            Check(index, true);
            return data.get()[index];
        }

        bool Check(size_t index, bool justThrow = false) const
        {
            if(index >= size || index < 0)
            {
                stringstream stream;
                stream << "Buffer Index " << index << " Out of Range [0, " << size << "]";
                if(justThrow)
                    throw runtime_error(stream.str());
                else
                    return false;
            }
            else return true;
        }

    private:
        cl_mem_flags flag;
        size_t size;
        shared_ptr<T> data;
    };
}
