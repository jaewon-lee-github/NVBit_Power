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

void myNvml::measure_init()
{
}

void myNvml::measure_fin()
{
    // cout << "Measure fin" << endl;
    fclose(ofile);
    checkCudaNvmlErrors(nvmlDeviceResetGpuLockedClocks(device));
    checkCudaNvmlErrors(nvmlShutdown());
}

void myNvml::measure_start(const char *k_name)
{
    if (start_flag == 0)
    {
        char temp[256];
        start_flag++;
        debug_printf("Measure start\n");
        debug_printf("freq_mode: %d\n", freq_mode);
        checkCudaNvmlErrors(nvmlInit());
        checkCudaNvmlErrors(nvmlDeviceGetHandleByIndex(TARGET_DEVICE, &device));
        sprintf(temp, "output_%d_%s_%dms.csv", freq_mode, bench_name, interval);
        ofile = fopen(temp, "w");
        fprintf(ofile, "Kernel,Timestamp,Freq,Power\n");
        // fseek(ofile, -1, SEEK_CUR);
    }
    // ofile = fopen(temp, "a");
    if (kernel_map.find(k_name) == kernel_map.end())
    {
        debug_printf("New kernel(%s) will be measured for power\n", k_name);
        strncpy(kernel_name, k_name, sizeof(kernel_name));
        num_call = 0;
        prev_energy = 0;
        prev_power = 0;
        kernel_map[kernel_name] = 1;
        CBT->start(interval, [this](void){
                this->measure_energy_thread();
                }
                );
    }
    else
    {
        debug_printf("Kernel(%s) is already measured for power\n", k_name);
    }
}

void myNvml::measure_stop()
{
    CBT->stop();
    // measure_fin();
}

void myNvml::measure_energy_thread()
{
    // unsigned long long energy = 0;
    // double powerDiff = 0;
    unsigned int gpu_clock = 0;
    num_call++;
    // Get GPU clock and change GPU clock
    checkCudaNvmlErrors(nvmlDeviceGetClockInfo(device, NVML_CLOCK_GRAPHICS, &gpu_clock));
    debug_printf("Current clock: %d\n", gpu_clock);
    // Get power Usage
    unsigned int powerUsage = 0;
    checkCudaNvmlErrors(nvmlDeviceGetPowerUsage(device, &powerUsage));
    debug_printf("Power usage: %u\n", powerUsage);
    /*
    checkCudaNvmlErrors(nvmlDeviceGetTotalEnergyConsumption(device, &energy));
    powerDiff = ((double)energy - (double)prev_energy) / (double)interval;
    if (powerDiff == 0)
        powerDiff = prev_power;
    else
    {
        prev_energy = energy;
        prev_power = powerDiff;
    }
    */
    // debug_printf("Total energy consumed: %llu\n", energy);
    // debug_printf("power from energy: %lf\n", powerDiff);

    if (freq_mode != 0 && num_call % bm->getResetPeriod() == 0)
    {
        unsigned int target_freq = bm->getFreq();
        if (target_freq == 0)
        {
            // retry to get freq after reset.
            bm->resetBinCounters();
            target_freq = bm->getFreq();
        }
        debug_printf("Freq will be changed to %u\n", target_freq);
        checkCudaNvmlErrors(nvmlDeviceSetGpuLockedClocks(device, target_freq, target_freq));
    }

    // Kernel,Timestamp,Freq,Power
    fprintf(ofile, "%s_%s,%u,%u,%u\n", bench_name, kernel_name, num_call, gpu_clock, powerUsage);
    // fprintf(ofile, "%u,%u,%f", gpu_clock, powerUsage, powerDiff);
    prev_power = powerUsage;
}
