#pragma once
// =============================================================================
// csv_log.h — optional machine-readable output for every C++ benchmark.
//
// Each benchmark accepts   --csv <path>   on the command line. If given, every
// measurement is also written as one CSV row (in addition to the printed
// tables), so python/summarize.py can build the results summary without
// anyone copying numbers by hand. Without --csv, add() does nothing.
//
// Columns: gpu, benchmark, experiment, impl, shape, dtype, ms, gflops, gbs, note
// (empty cell = not applicable). Explained in docs/09_benchmarking.md Part 2.
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

class CsvLog {
public:
    CsvLog(int argc, char** argv, const std::string& benchmark, const std::string& gpu)
        : benchmark_(benchmark), gpu_(gpu) {
        for (int i = 1; i + 1 < argc; ++i) {
            if (std::strcmp(argv[i], "--csv") == 0) {
                file_ = std::fopen(argv[i + 1], "w");
                if (file_ == nullptr) {
                    std::fprintf(stderr, "cannot open CSV file %s\n", argv[i + 1]);
                    std::exit(EXIT_FAILURE);
                }
                std::fprintf(file_, "gpu,benchmark,experiment,impl,shape,dtype,ms,gflops,gbs,note\n");
            }
        }
    }
    ~CsvLog() {
        if (file_ != nullptr) std::fclose(file_);
    }
    CsvLog(const CsvLog&) = delete;
    CsvLog& operator=(const CsvLog&) = delete;

    // Pass a negative number for a metric that does not apply.
    void add(const std::string& experiment, const std::string& impl, const std::string& shape,
             const std::string& dtype, double ms, double gflops = -1.0, double gbs = -1.0,
             const std::string& note = "") {
        if (file_ == nullptr) return;
        std::fprintf(file_, "\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",%s,%s,%s,\"%s\"\n", gpu_.c_str(),
                     benchmark_.c_str(), experiment.c_str(), impl.c_str(), shape.c_str(), dtype.c_str(),
                     number(ms).c_str(), number(gflops).c_str(), number(gbs).c_str(), note.c_str());
        std::fflush(file_);  // keep rows even if a later step crashes
    }

private:
    static std::string number(double v) {
        if (v < 0) return "";
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.6g", v);
        return buf;
    }

    std::FILE* file_ = nullptr;
    std::string benchmark_;
    std::string gpu_;
};

// "1024x768" style shape labels.
inline std::string dims(int a, int b) {
    return std::to_string(a) + "x" + std::to_string(b);
}
inline std::string dims(int a, int b, int c) {
    return dims(a, b) + "x" + std::to_string(c);
}
