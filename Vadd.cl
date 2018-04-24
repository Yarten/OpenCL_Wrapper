

__kernel void test(
    __global int * a,
    __local int * b,
     int c,
     __global const float * e,
     float f)
{

}

__kernel void vadd(
    __global float* c, __global const float* a, __global const float* b) {

   int row = get_global_id(0);
   int col = get_global_id(1);
   int NumOfCol = get_global_size(1);

    if(col % 3 == 0)
   c[row * NumOfCol + col] = a[row * NumOfCol + col] + b[row * NumOfCol + col];
   else if(col % 3 == 1)
    c[row * NumOfCol + col] = a[row * NumOfCol + col] - b[row * NumOfCol + col];
    else
     c[row * NumOfCol + col] = - a[row * NumOfCol + col] + b[row * NumOfCol + col];

}