#ifndef _UTILITY_
#define _UTILITY_

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "stdc++.h"

using namespace std;

template <typename T>
vector<tuple<size_t, T>> desTupVec(const T* arr, int n) {
    vector<tuple<size_t, T>> tv(n);
    for(size_t i = 0; i < n; i++)
    {
        tv[i] = tuple<size_t, T>(i, arr[i]);
    }
    sort(tv.begin(), tv.end(),
        [](const tuple<size_t, T>& a,
           const tuple<size_t, T>& b) {return (get<1>(a) > get<1>(b));});
    return tv;
}

#endif