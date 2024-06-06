#include <iostream>
#include <algorithm>
#include <vector>
#include <random>
#include "bin.h"
#include <cassert>

BinManager::BinManager(int start_frq, int end_frq, int distance, int sampling_interval, int reset_interval)
    : start_frq(start_frq), end_frq(end_frq), distance(distance), sampling_interval(sampling_interval), reset_interval(reset_interval)
{
    num_bin_types = (end_frq - start_frq) / distance + 1;
    debug_printf("Number of bin types: %d\n", num_bin_types);
    assert(("num_bin_type should be bigger than 1", num_bin_types > 1));

    sample_per_reset = reset_interval / sampling_interval;
    debug_printf("sample per reset: %d\n", sample_per_reset);

    step_in_bin = ((sample_per_reset - num_bin_types) * 2) / (pow(num_bin_types, 2) - num_bin_types);
    debug_printf("step in bin: %d\n", step_in_bin);
    assert(step_in_bin >= 1);
}

// Function to get the frequency of a randomly chosen bin with counter < maximumCounter
int BinManager::getFreq()
{
    if (freq_bins.empty())
    {
        debug_printf("No eligible bins found!\n");
        return -1;
    }
    int freq = freq_bins.back();
    freq_bins.pop_back();
    debug_printf("Selected Bin Frequency:%d\n", freq);
    return freq;
}

int BinManager::getResetPeriod()
{
    return sample_per_reset;
}

void BinManager::setBinCounters(int bin_policy)
{
    int start = 0;
    int end = 0;
    int step = 0;
    int bin_step = 0;
    int cur_height = 1;

    freq_bins.clear();
    if (bin_policy == BIN_POLICY::FLAT)
    {
        start = start_frq;
        end = end_frq;
        step = distance;
        bin_step = 0;
        cur_height = sample_per_reset / num_bin_types;
    }
    else if (bin_policy == BIN_POLICY::INCLINED)
    {
        start = start_frq;
        end = end_frq;
        step = distance;
        bin_step = step_in_bin;
        cur_height = 1;
    }
    else if (bin_policy == BIN_POLICY::DECLINED)
    {
        start = end_frq;
        end = start_frq;
        step = -distance;
        bin_step = step_in_bin;
        cur_height = 1;
    }

    int slot_cnt = 0;
    assert((start - end) % step == 0);
    for (int i = start; i != end; i += step)
    {
        for (int j = 0; j < cur_height; j++)
            freq_bins.push_back(i);
        slot_cnt += cur_height;
        debug_printf("Added size %d bin with frequency %d\n", cur_height, i);
        cur_height += bin_step;
    }

    for (int j = 0; j < sample_per_reset - slot_cnt; j++)
        freq_bins.push_back(end);
    debug_printf("Added size %d bin to the last bin with frequency %d\n", sample_per_reset - slot_cnt, end);

    std::random_shuffle(freq_bins.begin(), freq_bins.end());
}
