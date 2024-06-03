#ifndef _MYNVML_H_
#define _MYNVML_H_

#include "cbt.h"
#include <nvml.h>
#include <stdio.h>
#include <string.h>
#include "bin.h"
#include <iostream>
#include <unordered_map>
#include <fstream>
#include <cassert>

class myNvml
{
public:
    void measure_init();
    void measure_fin();
    void measure_start(const char *kernel_name);
    void measure_stop();
    void reset();
    void get_time();
    void measure_energy_thread();
    // benchmark name is given by environment varialbe "BENCH_NAME"
    myNvml();
    ~myNvml();

private:
    cudaEvent_t start, stop;
    int target_device;
    int _min_freq;
    int _max_freq;
    int _step_freq;
    int start_flag;
    int _sampling_interval;
    int _reset_interval;
    CallBackTimer *CBT;
    int cpu_model;
    unsigned int num_call;
    bool isFixed;
    nvmlDevice_t device;
    unsigned long long prev_energy;
    unsigned long long prev_power;
    unsigned long long total_power;
    unsigned long long prev_avg_power;
    int freq;
    char bench_name[128];
    char kernel_name[128];
    FILE *ofile;
    FILE *otfile;
    BinManager *bm;
    int _freq_mode;
    int _bin_policy;
    unordered_map<string, unsigned int> kernel_map;
};

#endif
