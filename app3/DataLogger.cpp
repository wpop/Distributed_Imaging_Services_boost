#include "DataLogger.h"
#include <iostream>

DataLogger::DataLogger() : db(nullptr) {
    if (sqlite3_open("features.db", &db) != SQLITE_OK) {
        std::cerr << "Can't open database: " << sqlite3_errmsg(db) << std::endl;
        db = nullptr;
    } else {
        const char* sql = "CREATE TABLE IF NOT EXISTS SIFT_FEATURES (ID INTEGER PRIMARY KEY AUTOINCREMENT, IMAGE_ROWS INT, IMAGE_COLS INT, IMAGE_TYPE INT, IMAGE_DATA BLOB, KEYPOINTS BLOB, DESCRIPTORS BLOB);";
        char* errMsg = nullptr;
        if (sqlite3_exec(db, sql, 0, 0, &errMsg) != SQLITE_OK) {
            std::cerr << "SQL error creating table: " << errMsg << std::endl;
            sqlite3_free(errMsg);
        }
    }
}

DataLogger::~DataLogger() {
    if (db) {
        sqlite3_close(db);
    }
}

std::vector<SIFTFeaturesIpc> DataLogger::get() {
    managed_shared_memory segment(open_only, "FeatureSharedMemory");
    SharedData* data = segment.find<SharedData>("SharedFeaturesData").first;
    std::vector<SIFTFeaturesIpc> features;

    {
        scoped_lock<named_mutex> lock(data->mutex);
        while (!data->featuresReady) {
            data->condition_features_ready.wait(lock);
        }
        // Copy data out of shared memory into local process memory
        features = data->features; // operator= uses the local allocator by default
        data->featuresReady = false;
        std::cout << "DataLogger: Received " << features.size() << " feature sets." << std::endl;
    }
    return features;
}

void DataLogger::updateTable(const std::vector<SIFTFeaturesIpc>& features) {
    if (!db) return;

    const char* sql = "INSERT INTO SIFT_FEATURES (IMAGE_ROWS, IMAGE_COLS, IMAGE_TYPE, IMAGE_DATA, KEYPOINTS, DESCRIPTORS) VALUES (?, ?, ?, ?, ?, ?);";
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        std::cerr << "Failed to prepare statement: " << sqlite3_errmsg(db) << std::endl;
        return;
    }

    for (const auto& ipcFeatures : features) {
        sqlite3_bind_int(stmt, 1, ipcFeatures.image.rows);
        sqlite3_bind_int(stmt, 2, ipcFeatures.image.cols);
        sqlite3_bind_int(stmt, 3, ipcFeatures.image.type);
        sqlite3_bind_blob(stmt, 4, ipcFeatures.image.data.data(), ipcFeatures.image.data.size(), SQLITE_STATIC);
        
        // Store keypoints as a blob (vector of bytes)
        size_t kp_size = ipcFeatures.keypoints.size() * sizeof(cv::KeyPoint);
        sqlite3_bind_blob(stmt, 5, ipcFeatures.keypoints.data(), kp_size, SQLITE_STATIC);

        // Store descriptors as a blob
        size_t desc_size = ipcFeatures.descriptors.data.size();
        sqlite3_bind_blob(stmt, 6, ipcFeatures.descriptors.data.data(), desc_size, SQLITE_STATIC);

        if (sqlite3_step(stmt) != SQLITE_DONE) {
            std::cerr << "Execution failed: " << sqlite3_errmsg(db) << std::endl;
        }
        sqlite3_reset(stmt);
    }

    sqlite3_finalize(stmt);
    std::cout << "DataLogger: Inserted " << features.size() << " records into SQLite table." << std::endl;
}
