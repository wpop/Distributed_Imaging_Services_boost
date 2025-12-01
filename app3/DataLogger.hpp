#ifndef DATALOGGER_H
#define DATALOGGER_H

#include "../lib/SharedMemory.hpp"
#include <sqlite3.h>
#include <vector>

class DataLogger {
private:
    sqlite3* db;
    std::vector<SIFTFeaturesIpc> get();
    void updateTable(const std::vector<SIFTFeaturesIpc>& features);
public:
    DataLogger();
    ~DataLogger();
};

#endif // DATALOGGER_H
