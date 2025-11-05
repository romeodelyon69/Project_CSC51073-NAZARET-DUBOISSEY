#include "Nlib.h"
#include <algorithm>
#include <numeric>

Buffer::Buffer(int size) : size(size) {
    data.reserve(size);
}

void Buffer::add(float value) {
    if (data.size() >= static_cast<size_t>(size)) {
        data.erase(data.begin());
    }
    data.push_back(value);
}

std::optional<float> Buffer::get_median() const {
    if (data.empty()) {
        return std::nullopt;
    }
    
    // Create a copy to sort (to keep the method const)
    std::vector<float> sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());
    
    return sorted_data[sorted_data.size() / 2];
}

std::optional<float> Buffer::get_mean() const {
    if (data.empty()) {
        return std::nullopt;
    }
    
    float sum = std::accumulate(data.begin(), data.end(), 0.0f);
    return sum / static_cast<float>(data.size());
}

