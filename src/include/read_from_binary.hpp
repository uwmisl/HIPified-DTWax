#ifndef READ_FROM_BINARY_HPP
#define READ_FROM_BINARY_HPP

#include "common.hpp"
#include "datatypes.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <stdexcept>

inline void readQueriesFromBin(const std::string& data_file, value_t* queries)
{
    std::ifstream queriesFile(data_file, std::ios::binary);
    if (!queriesFile)
    {
        throw std::runtime_error("Error: File '" + data_file + "' does not exist or cannot be opened.");
    }

    // Read the expected number of values
    size_t total_values = NUM_READS * QUERY_LEN;
    queriesFile.read(reinterpret_cast<char*>(queries), total_values * sizeof(value_t));
}
#endif