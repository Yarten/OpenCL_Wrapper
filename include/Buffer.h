//
// Created by yarten on 18-4-23.
//

#pragma once

#include <cstdlib>
#include <CL/cl.hpp>
#include <memory>
#include <exception>
#include <sstream>
#include <initializer_list>
#include <vector>

using std::shared_ptr;
using std::runtime_error;
using std::stringstream;
using std::initializer_list;
using std::vector;

namespace opencl
{
    template <typename T>
    class Buffer
    {
    public:
        /// 构造器
        explicit Buffer(size_t size = -1, cl_mem_flags flag = 0, T defaultValue = T())
                : size(size), flag(flag), weakData(nullptr)
        {
            if(size == -1)
            {
                data.reset(new T(defaultValue));
                this->flag = 0;
            }
            else
                Set(size, flag, defaultValue);
        }

        Buffer(initializer_list<size_t> dims, cl_mem_flags flag = 0, T defaultValue = T())
        {
            Set(dims, flag, defaultValue);
        }

        Buffer(initializer_list<size_t> dims, initializer_list<T> data, cl_mem_flags flag = 0)
        {
            Set(dims, flag);
            size_t index = 0;
            for(const T & i : data)
                this->data.get()[index++] = i;
        }

        Buffer(shared_ptr<T> data, size_t size, cl_mem_flags flag = 0)
        {
            Set(data, size, flag);
        }

        Buffer(shared_ptr<T> data, initializer_list<size_t> dims, cl_mem_flags flag = 0)
        {
            Set(data, dims, flag);
        }

        Buffer(T * data, size_t size, bool weak = false, cl_mem_flags flag = 0)
        {
            Set(data, size, weak, flag);
        }

        Buffer(T * data, initializer_list<size_t> dims, bool weak = false, cl_mem_flags flag = 0)
        {
            Set(data, dims, weak, flag);
        }

        /// 赋值接口
        void Set(size_t size, cl_mem_flags flag = -1, T defaultValue = T())
        {
            Set({size}, flag, defaultValue);
        }

        void Set(initializer_list<size_t> dims, cl_mem_flags flag = -1, T defaultValue = T())
        {
            Resize(dims);
            if(flag != -1)
                this->flag = flag;
            data.reset(new T[size]);
            for(int i = 0; i < size; i++)
                data.get()[i] = defaultValue;
            weakData = nullptr;
        }

        void Set(shared_ptr<T> data, size_t size, cl_mem_flags flag = -1)
        {
            Set(data, {size}, flag);
        }

        void Set(shared_ptr<T> data, initializer_list<size_t> dims, cl_mem_flags flag = -1)
        {
            Resize(dims);
            if(flag != -1)
                this->flag = flag;
            this->data = data;
            this->weakData = nullptr;
        }

        void Set(T * data, size_t size, bool weak = false, cl_mem_flags flag = -1)
        {
            Set(data, {size}, weak, flag);
        }

        void Set(T * data, initializer_list<size_t> dims, bool weak = false, cl_mem_flags flag  = -1)
        {
            Resize(dims);
            if(flag != -1)
                this->flag = flag;
            if(weak)
            {
                weakData = data;
                this->data.reset();
            }
            else
            {
                weakData = nullptr;
                this->data.reset(new T[size]);
                for(int i = 0; i < size; i++)
                    this->data.get()[i] = data[i];
            }
        }

        void Resize(initializer_list<size_t> dims)
        {
            size = 1;
            vector<size_t> temp = dims;
            for(size_t i : temp)
                size *= i;

            int dimnum = temp.size();
            this->dims.resize(dimnum);
            this->dims[dimnum-1] = 1;

            for(int i = dimnum-2; i >= 0; i--)
                this->dims[i] = this->dims[i+1] * temp[i+1];
        }

        /// 信息获取接口
        T * Data() { return weakData ? weakData : data.get(); }

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
                this->weakData = nullptr;
                data.reset(new T(value));
            }
            else
                *data = value;

            return *this;
        }

        operator T() const { return weakData ? *weakData : *data; }

        operator T&() { return weakData ? *weakData : *data; }

        /// 数组操作接口
        T & operator[](size_t index)
        {
            Check(index, true);
            return weakData ? weakData[index] : data.get()[index];
        }

        const T & operator[](size_t index) const
        {
            Check(index, true);
            return weakData ? weakData[index] : data.get()[index];
        }


        T & operator()(initializer_list<size_t> indexs)
        {
            size_t index = GetIndex(indexs);
            return operator[](index);
        }

        const T &operator()(initializer_list<size_t> indexs) const
        {
            size_t index = GetIndex(indexs);
            return operator[](index);
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

        size_t GetIndex(initializer_list<size_t> indexs) const
        {
            if(indexs.size() != dims.size())
            {
                stringstream stream;
                stream << "Dimension Wrong ! Use " << indexs.size() << " But Expected " << dims.size();
                throw runtime_error(stream.str());
            }

            size_t index = 0;
            int i = 0;
            for(size_t j : indexs)
            {
                index += dims[i++] * j;
            }

            return index;
        }

    private:
        cl_mem_flags flag;
        size_t size;
        std::vector<size_t> dims;
        shared_ptr<T> data;
        T * weakData;
    };
}
