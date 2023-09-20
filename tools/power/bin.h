#ifndef BINMANAGER_H
#define BINMANAGER_H

#include <vector>
#ifdef DEBUG
#define debug_printf(...) printf(__VA_ARGS__)
#else
#define debug_printf(...) ((void)0)
#endif

using namespace std;

class Bin
{
public:
    Bin(int freq, int maxCounter) : frequency(freq), maximumCounter(maxCounter), counter(0) {}

    unsigned int getFrequency();
    void incrementCounter();
    void resetCounter();
    bool is_full();

private:
    int frequency;
    int maximumCounter;
    int counter;
};

class BinManager
{
public:
    // Constructor to initialize the bins
    BinManager(int bin_size, int start_frq, int end_frq, int distance, int reset_period)
        : bin_size(bin_size), start_frq(start_frq), end_frq(end_frq), distance(distance), reset_period(reset_period)
    {
        for (int i = start_frq; i <= end_frq; i += distance)
        {
            addBin(i, bin_size);
            debug_printf("Added size %d bin with frequency %d\n", bin_size, i);
        }
    }
    // Add a bin to the manager
    void addBin(int freq, int maxCounter);
    int getResetPeriod();

    // Function to get the frequency of a randomly chosen bin with counter < maximumCounter
    unsigned int getFreq();
    void resetBinCounters();

private:
    int bin_size;
    int start_frq;
    int end_frq;
    int distance;
    int reset_period;
    std::vector<Bin> bins;
};

#endif // BINMANAGER_H