#ifndef CALLBACKTIMER_H
#define CALLBACKTIMER_H
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <inttypes.h>
#include <unistd.h>
#include <math.h>
#include <string.h>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <pthread.h>
#include <atomic>
#include <functional>
#include <chrono>

#include <sys/syscall.h>
// #include <linux/perf_event.h>

// using namespace std;

class CallBackTimer
{
public:
    CallBackTimer()
        : _execute(false)
    {
    }

    ~CallBackTimer()
    {
        if (_execute.load(std::memory_order_acquire))
        {
            stop();
        };
    }

    void stop();
    void start(int interval, std::function<void(void)> func);
    bool is_running() const noexcept;

private:
    std::atomic<bool> _execute;
    std::thread _thd;
};
#endif // CALLBACKTIMER_H
