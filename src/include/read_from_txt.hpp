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

inline void readRefFromTxt(std::string data_file, value_t *ref)
{
    std::ifstream refFile(data_file);
    if (!refFile)
    {
      throw std::runtime_error("Error: File '" + data_file + "' does not exist or cannot be opened.");
    }

    std::string line;

    // Reference file should contain one line of space separated values
    if (!std::getline(refFile, line)) {
        throw std::runtime_error("Error: Could not find reference data in file");
    }
    std::stringstream ss(line);
    float value;
    size_t index = 0;
    while (ss >> value) {
        if (index >= REF_LEN) {
            throw std::runtime_error("Error: Reference line contains more numbers than the expected REF_LEN=" + std::to_string(REF_LEN));
        }
        ref[index++] = float(value);
    }
    if (index != REF_LEN) {
        throw std::runtime_error("Error: Reference line contains fewer numbers than the expected REF_LEN=" + std::to_string(REF_LEN));
    }
}


inline void readQueriesFromTxt(std::string data_file, value_t *queries)
{
    std::ifstream queriesFile(data_file);
    if (!queriesFile)
    {
      throw std::runtime_error("Error: File '" + data_file + "' does not exist or cannot be opened.");
    }

    std::string line;
    size_t query_num = 0;
    while (std::getline(queriesFile, line)) {
        // Skip empty lines (e.g., trailing whitespace)
        if (line.empty()) continue;
        // Read each line of space separated values
        std::stringstream ss(line);
        float value;
        size_t index = 0;
        while (ss >> value) {
            if (index >= QUERY_LEN) {
                throw std::runtime_error("Error: Query line has more numbers than the expected QUERY_LEN=" + std::to_string(QUERY_LEN));
            }
            queries[query_num * QUERY_LEN + index++] = float(value);
        }
        if (index != QUERY_LEN) {
            throw std::runtime_error("Error: Query line has fewer numbers than the expected QUERY_LEN=" + std::to_string(QUERY_LEN));
        }
        query_num++;
    }

    // Validate the number of query lines
    if (query_num != NUM_READS) {
        throw std::runtime_error("Error: Number of query lines found in file does not match the expected NUM_READS=" + std::to_string(NUM_READS));
    }
}

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
                throw std::runtime_error("Error: Reference line has more numbers than REF_LEN=" + std::to_string(REF_LEN));
            }
            ref[index++] = FLOAT2HALF(value);
        }
        if (index != REF_LEN) {
            throw std::runtime_error("Error: Reference line has fewer numbers than REF_LEN=" + std::to_string(REF_LEN));
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
                throw std::runtime_error("Error: Query line has more numbers than QUERY_LEN=" + std::to_string(QUERY_LEN));
            }
            queries[query_row * QUERY_LEN + index++] = FLOAT2HALF(value);
        }
        if (index != QUERY_LEN) {
            throw std::runtime_error("Error: Query line has fewer numbers than QUERY_LEN=" + std::to_string(QUERY_LEN));
        }
        query_row++;
    }

    // Validate the number of query lines
    if (query_row != NUM_READS) {
        throw std::runtime_error("Error: Number of query lines does not match NUM_READS=" + std::to_string(NUM_READS));
    }
}

#endif