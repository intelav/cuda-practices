#include <iostream>
#include <helper_cuda.h>
#include "range.hpp"

using namespace util::lang;

template <typename T>
using step_range = typename range_proxy<T>::step_range_proxy;



