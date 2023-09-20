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
#define TARGET_DEVICE 0

class myNvml
{ 
public:
    void measure_init();
    void measure_fin();
    void measure_start(const char *kernel_name);
    void measure_stop();
    void measure_energy_thread();
    // benchmark name is given by environment varialbe "BENCH_NAME"
    myNvml(int mode, int interval)
    {
        const char *envVarValue = std::getenv("BENCH_NAME");
        debug_printf("c_str:BENCH_NAME= %s\n", envVarValue);
        if (envVarValue != NULL)
        {
            strncpy(bench_name, envVarValue, sizeof(bench_name));
        }
        else
        {
            strncpy(bench_name, "unknown", sizeof(bench_name));
        }
        freq_mode = mode;
        start_flag = 0;
        CBT = new CallBackTimer();
        bm = new BinManager(1, 500, 2000, 100, 1);
        this->interval = interval;
        num_call = 0;
        prev_energy = 0;
        prev_power = 0;
    }

    ~myNvml()
    {
        delete CBT;
        delete bm;
    }

private:
    int start_flag;
    int interval;
    CallBackTimer *CBT;
    int cpu_model;
    unsigned int num_call;
    nvmlDevice_t device;
    unsigned long long prev_energy;
    unsigned long long prev_power;
    int freq;
    char bench_name[128];
    char kernel_name[128];
    FILE *ofile;
    BinManager *bm;
    int freq_mode;
    unordered_map<string, unsigned int> kernel_map;
};

#endif