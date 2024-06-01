#ifndef BINMANAGER_H
#define BINMANAGER_H

#include <vector>
#ifdef DEBUG
#define debug_printf(...) printf(__VA_ARGS__)
#else
#define debug_printf(...) ((void)0)
#endif

using namespace std;

enum FREQ_MODE
{
    ORG,
    FIXED,
    RANDOM,
    ADAPTIVE 
};
enum BIN_POLICY
{
    FLAT,
    INCLINED,
    DECLINED
};

class BinManager
{
public:
    BinManager(int start_frq, int end_frq, int distance, int sampling_interval, int reset_interval);
    // Add a bin to the manager
    // void addBin(int freq, int maxCounter);
    int getResetPeriod();
    void printBins();
    // Function to get the frequency of a randomly chosen bin with counter < maximumCounter
    int getFreq(); // returns -1 if no bin is available
    void setBinCounters(int bin_policy);
    // void resetBinCounters();
    // void updateMaxCap(unsigned int mode);

private:
    int freq_mode;
    int bin_policy;
    int bin_max_cnt;
    int bin_size;
    int start_frq;
    int end_frq;
    int distance;
    int _max_height;
    int sampling_interval;
    int reset_interval;
    int num_bin_types;
    int sample_per_reset;
    int step_in_bin;
    int _reset_period;
    // std::vector<Bin> bins;
    std::vector<int> freq_bins;
};

#endif // BINMANAGER_H
