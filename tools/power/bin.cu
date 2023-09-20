#include <iostream>
#include <vector>
#include <random>
#include "bin.h"

inline unsigned int Bin::getFrequency()
{
    if (!is_full())
    {
        return frequency;
    }
    return 0;
}

inline void Bin::incrementCounter()
{
    if (!is_full())
    {
        debug_printf("Incrementing counter for bin with frequency:%d\n", frequency);
        counter++;
    }
}

inline void Bin::resetCounter()
{
    debug_printf("Resetting counter for bin with %d\n", frequency);
    counter = 0;
}

inline bool Bin::is_full()
{
    debug_printf("Counter: %d\n", counter);
    debug_printf("Maximum Counter: %d\n", maximumCounter);
    return (counter == maximumCounter);
}

void BinManager::addBin(int freq, int maxCounter)
{
    bins.emplace_back(freq, maxCounter);
}

// Function to get the frequency of a randomly chosen bin with counter < maximumCounter
unsigned int BinManager::getFreq()
{
    // Collect eligible bins
    std::vector<int> eligibleBins;
    for (long unsigned int i = 0; i < bins.size(); i++)
    {
        if (!bins[i].is_full())
        {
            // cout << "Bin " << i << " is eligible" << endl;
            eligibleBins.push_back(i);
        }
        // else
        // cout << "Bin " << i << " is uneligible" << endl;
    }
    if (eligibleBins.empty())
    {
        debug_printf("No eligible bins found!\n");
        return 0;
    }

    // Randomly choose a bin from eligible bins
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, eligibleBins.size() - 1);

    int selectedBinIndex = eligibleBins[dist(gen)];
    int selectedBinFreq = bins[selectedBinIndex].getFrequency();
    debug_printf("Selected Bin Index:%d\n", selectedBinIndex);
    debug_printf("Selected Bin Frequency:%d\n", selectedBinFreq);

    // Increment the counter of the selected bin
    bins[selectedBinIndex].incrementCounter();

    return selectedBinFreq;
}

int BinManager::getResetPeriod()
{
    return reset_period;
}

void BinManager::resetBinCounters()
{
    // cout << "Resetting bin counters..." << endl;
    for (long unsigned int i = 0; i < bins.size(); i++)
    {
        bins[i].resetCounter();
    }
}