#ifndef READ_FROM_TXT_HPP
#define READ_FROM_TXT_HPP

#include "common.hpp"
#include "datatypes.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream> // for stringstream

inline void readDataFromTxt(std::ifstream &inFile, value_t *ref, value_t *queries, index_t num_queries)
{
    std::vector<std::vector<float>> data;
    std::string line;

    // The first line is the reference
    std::getline(inFile, line);
    {
        std::stringstream ss(line);
        float value;
        size_t index = 0;
        while (ss >> value)
        {
            ref[index++] = FLOAT2HALF(value);
        }
    }

    // The subsequent lines are the queries
    size_t query_row = 0;
    while (std::getline(inFile, line))
    {
        std::stringstream ss(line);
        float value;
        size_t index = 0;
        while (ss >> value)
        {
            queries[query_row * num_queries + index++] = FLOAT2HALF(value);
        }
        query_row++;
    }
}

#endif
