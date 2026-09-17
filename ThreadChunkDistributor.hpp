#pragma once

#include <vector>
#include <algorithm>
#include <cstddef>

using namespace std;

struct ChunkRange {
    size_t start;
    size_t end;
};


class ThreadChunkDistributor {
public:
    
    static vector<ChunkRange> distribute(size_t totalElements, size_t requestedThreads) {
        if (totalElements == 0 || requestedThreads == 0) {
            return {};
        }

        size_t actualThreads = min(totalElements, requestedThreads);
        if (actualThreads == 0) actualThreads = 1;

        vector<ChunkRange> chunks;
        chunks.reserve(actualThreads);

        size_t baseChunkSize = totalElements / actualThreads;
        size_t remainder = totalElements % actualThreads;

        size_t currentStart = 0;
        for (size_t i = 0; i < actualThreads; ++i) {
            size_t chunkSize = baseChunkSize + (i < remainder ? 1 : 0);
            size_t currentEnd = currentStart + chunkSize;
            chunks.push_back({ currentStart, currentEnd });
            currentStart = currentEnd;
        }

        return chunks;
    }
};
