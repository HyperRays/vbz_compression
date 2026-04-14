#include "vbz.h"
#include "../test/test_utils.h"

#include <hdf5.h>
#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

static std::vector<std::vector<int16_t>> g_signals;

// Callback for iterating root groups of a multi-read fast5.
// Each root group is a read; signal lives at <read_id>/Raw/Signal.
static herr_t visit_multiread_group(
    hid_t loc_id, const char* name, const H5L_info_t*, void* op_data)
{
    auto* signals = static_cast<std::vector<std::vector<int16_t>>*>(op_data);
    std::string signal_path = std::string(name) + "/Raw/Signal";
    if (H5Lexists(loc_id, signal_path.c_str(), H5P_DEFAULT) > 0) {
        auto signal = read_1d_dataset<int16_t>(loc_id, signal_path.c_str(), H5T_NATIVE_INT16);
        if (!signal.empty()) {
            signals->push_back(std::move(signal));
        }
    }
    return 0; // continue
}

// Callback for iterating /Raw/Reads groups of a single-read fast5.
// Each group is Read_N; signal lives at <Read_N>/Signal.
static herr_t visit_singleread_group(
    hid_t loc_id, const char* name, const H5L_info_t*, void* op_data)
{
    auto* signals = static_cast<std::vector<std::vector<int16_t>>*>(op_data);
    std::string signal_path = std::string(name) + "/Signal";
    if (H5Lexists(loc_id, signal_path.c_str(), H5P_DEFAULT) > 0) {
        auto signal = read_1d_dataset<int16_t>(loc_id, signal_path.c_str(), H5T_NATIVE_INT16);
        if (!signal.empty()) {
            signals->push_back(std::move(signal));
        }
    }
    return 0; // continue
}

static bool load_signals_from_fast5(const std::string& path)
{
    hid_t file_id = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        std::cerr << "ERROR: Could not open: " << path << "\n";
        return false;
    }

    // Try multi-read fast5 first (modern format).
    H5Literate(file_id, H5_INDEX_NAME, H5_ITER_NATIVE, nullptr,
               visit_multiread_group, &g_signals);

    // Fallback: single-read fast5 (legacy format) — /Raw/Reads/Read_N/Signal.
    if (g_signals.empty() && H5Lexists(file_id, "/Raw/Reads", H5P_DEFAULT) > 0) {
        hid_t reads_gid = H5Gopen(file_id, "/Raw/Reads", H5P_DEFAULT);
        if (reads_gid >= 0) {
            H5Literate(reads_gid, H5_INDEX_NAME, H5_ITER_NATIVE, nullptr,
                       visit_singleread_group, &g_signals);
            H5Gclose(reads_gid);
        }
    }

    H5Fclose(file_id);
    return !g_signals.empty();
}

static void fast5_compress_benchmark(benchmark::State& state)
{
    if (g_signals.empty()) {
        state.SkipWithError("No signal data loaded");
        return;
    }

    std::size_t max_samples = 0;
    for (auto const& s : g_signals) {
        max_samples = std::max(max_samples, s.size());
    }

    CompressionOptions options{
        true,             // perform_delta_zig_zag — matches default VBZ settings
        sizeof(int16_t),  // integer_size
        1,                // zstd_compression_level
        VBZ_DEFAULT_VERSION
    };

    std::vector<char> dest(
        vbz_max_compressed_size(vbz_size_t(max_samples * sizeof(int16_t)), &options));

    std::size_t total_samples = 0;
    for (auto _ : state) {
        total_samples = 0;
        for (auto const& signal : g_signals) {
            auto const byte_count = signal.size() * sizeof(int16_t);
            dest.resize(dest.capacity());
            auto compressed_size = vbz_compress(
                signal.data(),
                vbz_size_t(byte_count),
                dest.data(),
                vbz_size_t(dest.size()),
                &options);
            benchmark::DoNotOptimize(compressed_size);
            total_samples += signal.size();
        }
    }

    state.SetItemsProcessed(int64_t(state.iterations()) * int64_t(total_samples));
    state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(total_samples) * int64_t(sizeof(int16_t)));
}

BENCHMARK(fast5_compress_benchmark);

int main(int argc, char** argv)
{
    std::string fast5_path;
    std::vector<char*> remaining;
    remaining.push_back(argv[0]);

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        const std::string prefix = "--fast5_file=";
        if (arg.compare(0, prefix.size(), prefix) == 0) {
            fast5_path = arg.substr(prefix.size());
        } else {
            remaining.push_back(argv[i]);
        }
    }

    if (fast5_path.empty()) {
        std::cerr << "Usage: " << argv[0]
                  << " --fast5_file=<path.fast5> [benchmark options]\n";
        return 1;
    }

    std::cerr << "Loading signals from: " << fast5_path << "\n";
    if (!load_signals_from_fast5(fast5_path)) {
        std::cerr << "ERROR: No signal data found in " << fast5_path << "\n";
        return 1;
    }
    std::cerr << "Loaded " << g_signals.size() << " reads\n";

    int remaining_argc = int(remaining.size());
    benchmark::Initialize(&remaining_argc, remaining.data());
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
