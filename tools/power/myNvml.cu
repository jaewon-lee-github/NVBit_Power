#include <cstdlib>
#include "cbt.h"
#include "myNvml.h"
#include <sys/time.h>

using namespace std;
template <typename T>
void check(T result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
                static_cast<unsigned int>(result), nvmlErrorString(result), func);
        nvmlShutdown();
        exit(EXIT_FAILURE);
    }
}
#define checkCudaNvmlErrors(val) check((val), #val, __FILE__, __LINE__)

myNvml::myNvml()
{
    if (getenv("BENCH_NAME") != NULL)
        strncpy(bench_name, getenv("BENCH_NAME"), sizeof(bench_name));
    else
        strncpy(bench_name, "unknown", sizeof(bench_name));
    debug_printf("BENCH_NAME= %s\n", bench_name);

    if (getenv("DEVICES_ID") != NULL)
        target_device = std::atoi(getenv("DEVICES_ID"));
    else
        target_device = 0;
    debug_printf("device_id = %d\n", target_device);

    if (getenv("FREQ_MODE") != NULL)
        _freq_mode = std::atoi(getenv("FREQ_MODE"));
    else
        _freq_mode = 0;
    debug_printf("freq_mode= %d\n", _freq_mode);

    if (getenv("BIN_POLICY") != NULL)
        _bin_policy = std::atoi(getenv("BIN_POLICY"));
    else
        _bin_policy = 0;
    debug_printf("bin_policy= %d\n", _bin_policy);

    if (getenv("MIN_FREQ") != NULL)
        _min_freq = std::atoi(getenv("MIN_FREQ"));
    else
        _min_freq = 500;
    debug_printf("min_freqw= %d\n", _min_freq);

    if (getenv("MAX_FREQ") != NULL)
        _max_freq = std::atoi(getenv("MAX_FREQ"));
    else
        _max_freq = 2000;
    debug_printf("max_freqw= %d\n", _max_freq);

    if (getenv("STEP_FREQ") != NULL)
        _step_freq = std::atoi(getenv("STEP_FREQ"));
    else
        _step_freq = 500;
    debug_printf("step_freq= %d\n", _step_freq);

    if (getenv("SAMPLING_INTERVAL") != NULL)
        _sampling_interval = std::atoi(getenv("SAMPLING_INTERVAL"));
    else
        _sampling_interval = 50;
    debug_printf("sampling interval= %d\n", _sampling_interval);

    if (getenv("RESET_INTERVAL") != NULL)
        _reset_interval = std::atoi(getenv("RESET_INTERVAL"));
    else
        _reset_interval = 2000;
    debug_printf("reset_interval= %d\n", _reset_interval);

    CBT = new CallBackTimer();
    bm = new BinManager(_min_freq, _max_freq, _step_freq, _sampling_interval, _reset_interval);
    num_call = 0;
    prev_energy = 0;
    prev_power = 0;
    prev_avg_power = 0;
    total_power = 0;
    isFixed = false;
    debug_printf("myNvml constructor Ends\n");
}
myNvml::~myNvml()
{
    // cudaEventDestroy(start);
    // cudaEventDestroy(stop);
    delete CBT;
    delete bm;
}

void myNvml::measure_init()
{
    // cudaEventRecord(start);
}

void myNvml::measure_fin()
{
    debug_printf("Measure fin\n");
    fclose(ofile);
    checkCudaNvmlErrors(nvmlDeviceResetGpuLockedClocks(device));
    nvmlShutdown();
}

// We will measured power only once for kernel
//

void myNvml::reset()
{
    num_call = 0;
    prev_energy = 0;
    prev_power = 0;
    total_power = 0;
    prev_avg_power = 0;
}

void myNvml::measure_start(const char *k_name)
{
    if (start_flag == 0)
    {
        char temp[256];
        start_flag++;
        debug_printf("Measure start\n");
        debug_printf("freq_mode: %d\n", _freq_mode);
        checkCudaNvmlErrors(nvmlInit());
        checkCudaNvmlErrors(nvmlDeviceGetHandleByIndex(target_device, &device));
        sprintf(temp, "output_%d_%d_%d_%d_%d_%d_%s_%d_%d.csv",
                target_device, _freq_mode, _bin_policy, _min_freq, _max_freq, _step_freq,
                bench_name, _sampling_interval, _reset_interval);
        ofile = fopen(temp, "w");
        // otfile = fopen("time.csv", "w");
        reset();
        fprintf(ofile, "Benchmark,Kernel,Timestamp,Freq,FreqMode,BinPolicy,Power\n");
        // fseek(ofile, -1, SEEK_CUR);
    }
    if (_bin_policy == 10) // FIXME
    {
        strncpy(kernel_name, k_name, sizeof(kernel_name));
        debug_printf("Measure the power of kernel %s\n", kernel_name);
        CBT->start(_sampling_interval, [this](void)
                   { this->measure_energy_thread(); });
    }
    else
    {
        if (kernel_map.find(k_name) == kernel_map.end())
        {
            strncpy(kernel_name, k_name, sizeof(kernel_name));
            debug_printf("Newly executed kernel(%s) will be measured for power\n", kernel_name);
            reset();
            kernel_map[kernel_name] = 1;
            CBT->start(_sampling_interval, [this](void)
                       { this->measure_energy_thread(); });
        }
        else
        {
            debug_printf("Power of Kernel(%s) is already measured \n", k_name);
        }
    }
}

void myNvml::measure_stop()
{
    debug_printf("measure_stop(%s)\n", __func__);
    CBT->stop();
    // float milliseconds = 0;
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // fprintf(otfile, "%f\n", milliseconds);
    // measure_fin();
}

void myNvml::measure_energy_thread()
{
    // unsigned long long energy = 0;
    // double powerDiff = 0;
    unsigned int gpu_clock = 0;
    // Get GPU clock and change GPU clock
    checkCudaNvmlErrors(nvmlDeviceGetClockInfo(device, NVML_CLOCK_GRAPHICS, &gpu_clock));
    //  Get power Usage
    unsigned int powerUsage = 0;
    checkCudaNvmlErrors(nvmlDeviceGetPowerUsage(device, &powerUsage));
    total_power += powerUsage;

    // unsigned long long energy = 0;
    // checkCudaNvmlErrors(nvmlDeviceGetTotalEnergyConsumption(device, &energy));
    // powerDiff = ((double)energy - (double)prev_energy) / (double)sampling_interval;
    // if (powerDiff == 0)
    //     powerDiff = prev_power;
    // else
    // {
    //     prev_energy = energy;
    //     prev_power = powerDiff;
    // }
    // debug_printf("Total energy consumed: %llu\n", energy);
    // debug_printf("power from energy: %lf\n", powerDiff);

    // random bin frequency mode
    if (_freq_mode == FREQ_MODE::ORG)
    {
        // Nothing
    }
    // else if (_freq_mode == FREQ_MODE::FIXED && isFixed == false)
    else if (_freq_mode == FREQ_MODE::FIXED)
    {
        unsigned int target_freq = bm->getFreq();
        debug_printf("Freq will be fixed to %u\n", target_freq);
        checkCudaNvmlErrors(nvmlDeviceSetGpuLockedClocks(device, target_freq, target_freq));
        // isFixed = true;
    }
    else
    {
        debug_printf("[%d][%d] Current clock: %d\n", num_call, _freq_mode, gpu_clock);
        debug_printf("Power usage: %u\n", powerUsage);
        // num_call = 0 @init time
        if (num_call % bm->getResetPeriod() == 0)
        {
            debug_printf("Reset period\n");
            unsigned long long avg_power = total_power / bm->getResetPeriod();
            if (_freq_mode == FREQ_MODE::RANDOM)
                bm->setBinCounters(BIN_POLICY::FLAT);
            else if (_freq_mode == FREQ_MODE::ADAPTIVE)
            {
                if (avg_power < prev_avg_power) // prev_avg_power == 0 @init
                {
                    debug_printf("BIN_POLICY = INCLINED (%llu < %llu)\n", avg_power, prev_avg_power);
                    bm->setBinCounters(BIN_POLICY::INCLINED);
                }
                else
                {
                    debug_printf("BIN_POLICY = DECLINED (%llu >= %llu)\n", avg_power, prev_avg_power);
                    bm->setBinCounters(BIN_POLICY::DECLINED);
                }
            }
            prev_avg_power = avg_power;
            total_power = 0;
        }
        unsigned int target_freq = bm->getFreq();
        debug_printf("Freq will be changed to %u\n", target_freq);
        checkCudaNvmlErrors(nvmlDeviceSetGpuLockedClocks(device, target_freq, target_freq));
    }

    // Kernel,Timestamp,Freq,Power
    fprintf(ofile, "%s,%s,%u,%u,%u,%u,%u\n", bench_name, kernel_name, num_call, gpu_clock, _freq_mode, _bin_policy, powerUsage);
    // fprintf(ofile, "%u,%u,%f", gpu_clock, powerUsage, powerDiff);
    prev_power = powerUsage;
    num_call++;
}
