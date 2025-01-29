#ifndef READ_FROM_TXT_HPP
#define READ_FROM_TXT_HPP

#include "common.hpp"
#include "datatypes.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <stdexcept>

inline void readDataFromTxt(std::ifstream &inFile, value_t *ref, value_t *queries)
{
    std::string line;

    // The first line is the reference
    if (!std::getline(inFile, line)) {
        throw std::runtime_error("Error: Missing reference line");
    }
    {
        std::stringstream ss(line);
        float value;
        size_t index = 0;
        while (ss >> value) {
            if (index >= REF_LEN) {
                throw std::runtime_error("Error: Reference line has more numbers than REF_LEN");
            }
            ref[index++] = FLOAT2HALF(value);
        }
        if (index != REF_LEN) {
            throw std::runtime_error("Error: Reference line has fewer numbers than REF_LEN");
        }
    }

    // The subsequent lines are the queries
    size_t query_row = 0;
    while (std::getline(inFile, line)) {
        // Skip empty lines (e.g., trailing whitespace)
        if (line.empty()) continue;

        std::stringstream ss(line);
        float value;
        size_t index = 0;
        while (ss >> value) {
            if (index >= QUERY_LEN) {
                throw std::runtime_error("Error: Query line has more numbers than QUERY_LEN");
            }
            queries[query_row * QUERY_LEN + index++] = FLOAT2HALF(value);
        }
        if (index != QUERY_LEN) {
            throw std::runtime_error("Error: Query line has fewer numbers than QUERY_LEN");
        }
        query_row++;
    }

    // Validate the number of query lines
    if (query_row != NUM_READS) {
        throw std::runtime_error("Error: Number of query lines does not match NUM_READS");
    }
}

#endif