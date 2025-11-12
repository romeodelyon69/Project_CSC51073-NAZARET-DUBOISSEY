#ifndef NLIB_H
#define NLIB_H

#include <vector>
#include <optional>

class Buffer {
public:
    // Constructor
    explicit Buffer(int size);
    
    // Add a value to the buffer
    void add(float value);
    
    // Get the median value from the buffer
    std::optional<float> get_median() const;
    
    // Get the mean value from the buffer
    std::optional<float> get_mean() const;
    
    // Get the current size of data
    size_t get_data_size() const { return data.size(); }
    
private:
    int size;
    std::vector<float> data;
};

#endif // NLIB_H

