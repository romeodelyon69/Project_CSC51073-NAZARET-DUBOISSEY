#ifndef NLIB_H
#define NLIB_H

#include <vector>
#include <algorithm>
#include <stdexcept>

namespace nlib {

// Buffer class for storing and computing statistics on recent values
class Buffer {
public:
    explicit Buffer(size_t size);
    
    void add(double value);
    double getMedian() const;
    double getMean() const;
    size_t size() const;
    bool empty() const;
    void clear();

private:
    size_t maxSize_;
    std::vector<double> data_;
};

} // namespace nlib

#endif // NLIB_H

