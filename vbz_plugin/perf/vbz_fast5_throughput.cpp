// vbz_fast5_throughput.cpp
//
// Profiles VBZ compression throughput on real fast5 data.
// Designed for large datasets on Slurm with 1TB+ RAM:
//   - Phase 1: load all signal data from fast5 files into RAM (checkpointed per-file)
//   - Phase 2: timed compression pass over all in-memory data (no I/O in the hot path)
//
// Usage:
//   vbz_fast5_throughput --fast5_list=files.txt --checkpoint=run.ckpt --log=run.log
//   vbz_fast5_throughput --fast5_dir=/data/reads/ --checkpoint=run.ckpt --log=run.log
//
// Checkpointing:
//   On first run a checkpoint file is created. If the job is killed and restarted
//   with the same --checkpoint path, loading resumes from the last completed file.
//
// Signal handling:
//   SIGTERM / SIGINT flush the checkpoint and write a partial summary before exiting.

#include "vbz.h"
#include "../test/test_utils.h"

#include <hdf5.h>

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

// ---------------------------------------------------------------------------
// Signal handling
// ---------------------------------------------------------------------------

static volatile std::sig_atomic_t g_interrupted = 0;
static void on_signal(int) { g_interrupted = 1; }

// ---------------------------------------------------------------------------
// fast5 reading (same logic as vbz_fast5_perf.cpp)
// ---------------------------------------------------------------------------

static herr_t visit_multiread_group(
    hid_t loc_id, const char* name, const H5L_info_t*, void* op_data)
{
    auto* signals = static_cast<std::vector<std::vector<int16_t>>*>(op_data);
    std::string signal_path = std::string(name) + "/Raw/Signal";
    if (H5Lexists(loc_id, signal_path.c_str(), H5P_DEFAULT) > 0) {
        auto signal = read_1d_dataset<int16_t>(loc_id, signal_path.c_str(), H5T_NATIVE_INT16);
        if (!signal.empty())
            signals->push_back(std::move(signal));
    }
    return 0;
}

static herr_t visit_singleread_group(
    hid_t loc_id, const char* name, const H5L_info_t*, void* op_data)
{
    auto* signals = static_cast<std::vector<std::vector<int16_t>>*>(op_data);
    std::string signal_path = std::string(name) + "/Signal";
    if (H5Lexists(loc_id, signal_path.c_str(), H5P_DEFAULT) > 0) {
        auto signal = read_1d_dataset<int16_t>(loc_id, signal_path.c_str(), H5T_NATIVE_INT16);
        if (!signal.empty())
            signals->push_back(std::move(signal));
    }
    return 0;
}

// Returns number of reads appended to `out`. Returns 0 on failure.
static std::size_t load_fast5(const std::string& path,
                               std::vector<std::vector<int16_t>>& out)
{
    hid_t file_id = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0)
        return 0;

    std::size_t before = out.size();

    H5Literate(file_id, H5_INDEX_NAME, H5_ITER_NATIVE, nullptr,
               visit_multiread_group, &out);

    if (out.size() == before &&
        H5Lexists(file_id, "/Raw/Reads", H5P_DEFAULT) > 0) {
        hid_t reads_gid = H5Gopen(file_id, "/Raw/Reads", H5P_DEFAULT);
        if (reads_gid >= 0) {
            H5Literate(reads_gid, H5_INDEX_NAME, H5_ITER_NATIVE, nullptr,
                       visit_singleread_group, &out);
            H5Gclose(reads_gid);
        }
    }

    H5Fclose(file_id);
    return out.size() - before;
}

// ---------------------------------------------------------------------------
// File discovery
// ---------------------------------------------------------------------------

static void collect_fast5_from_dir(const std::string& dir,
                                    std::vector<std::string>& out)
{
    DIR* d = opendir(dir.c_str());
    if (!d) {
        std::cerr << "WARNING: cannot open directory: " << dir << "\n";
        return;
    }
    std::vector<std::string> subdirs;
    struct dirent* entry;
    while ((entry = readdir(d)) != nullptr) {
        std::string name(entry->d_name);
        if (name == "." || name == "..") continue;
        std::string full = dir + "/" + name;
        struct stat st;
        if (stat(full.c_str(), &st) != 0) continue;
        if (S_ISDIR(st.st_mode)) {
            subdirs.push_back(full);
        } else if (name.size() > 6 &&
                   name.substr(name.size() - 6) == ".fast5") {
            out.push_back(full);
        }
    }
    closedir(d);
    for (auto const& sub : subdirs)
        collect_fast5_from_dir(sub, out);
}

static std::vector<std::string> discover_files(const std::string& fast5_dir,
                                                 const std::string& fast5_list)
{
    std::vector<std::string> files;
    if (!fast5_dir.empty()) {
        collect_fast5_from_dir(fast5_dir, files);
    } else {
        std::ifstream f(fast5_list);
        std::string line;
        while (std::getline(f, line)) {
            if (!line.empty()) files.push_back(line);
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

// ---------------------------------------------------------------------------
// Checkpoint
// ---------------------------------------------------------------------------

struct Checkpoint {
    std::size_t files_loaded = 0; // how many files from sorted list are in RAM
};

static Checkpoint load_checkpoint(const std::string& path)
{
    Checkpoint ckpt;
    std::ifstream f(path);
    if (!f.is_open()) return ckpt;
    std::string key;
    std::size_t val;
    while (f >> key >> val) {
        if (key == "files_loaded") ckpt.files_loaded = val;
    }
    return ckpt;
}

static void save_checkpoint(const std::string& path, const Checkpoint& ckpt)
{
    std::ofstream f(path);
    f << "files_loaded " << ckpt.files_loaded << "\n";
}

// ---------------------------------------------------------------------------
// Logging helpers
// ---------------------------------------------------------------------------

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

static std::string now_iso()
{
    auto t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", std::localtime(&t));
    return buf;
}

static double to_mbs(uint64_t bytes, double ns)
{
    if (ns <= 0.0) return 0.0;
    return double(bytes) / 1e6 / (ns / 1e9);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv)
{
    std::signal(SIGTERM, on_signal);
    std::signal(SIGINT,  on_signal);

    // Suppress HDF5 automatic error printing — we check return values ourselves.
    H5Eset_auto2(H5E_DEFAULT, NULL, NULL);

    // ---- Parse arguments ---------------------------------------------------
    std::string fast5_dir, fast5_list, checkpoint_path, log_path;
    std::size_t checkpoint_interval = 100; // files between checkpoint writes
    double      log_interval_s      = 30.0; // seconds between progress logs
    bool        dry_run             = false;
    std::size_t dry_run_files       = 10;   // files to sample in dry run

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--dry-run") { dry_run = true; continue; }
        auto eq = arg.find('=');
        if (eq == std::string::npos) continue;
        std::string key = arg.substr(2, eq - 2); // strip leading --
        std::string val = arg.substr(eq + 1);
        if      (key == "fast5_dir")           fast5_dir           = val;
        else if (key == "fast5_list")          fast5_list          = val;
        else if (key == "checkpoint")          checkpoint_path     = val;
        else if (key == "log")                 log_path            = val;
        else if (key == "checkpoint_interval") checkpoint_interval = std::stoul(val);
        else if (key == "log_interval")        log_interval_s      = std::stod(val);
        else if (key == "dry-run-files")       dry_run_files       = std::stoul(val);
    }

    if (fast5_dir.empty() && fast5_list.empty()) {
        std::cerr << "Usage: " << argv[0] << "\n"
                  << "  --fast5_dir=<dir>  OR  --fast5_list=<file>\n"
                  << "  --checkpoint=<file>   (default: vbz_throughput.ckpt)\n"
                  << "  --log=<file>          (default: vbz_throughput.log)\n"
                  << "  --checkpoint_interval=<N>  files between checkpoints (default 100)\n"
                  << "  --log_interval=<seconds>   progress log interval (default 30)\n"
                  << "  --dry-run               sample a few files and estimate full run\n"
                  << "  --dry-run-files=<N>     files to sample in dry run (default 10)\n";
        return 1;
    }

    if (checkpoint_path.empty()) checkpoint_path = "vbz_throughput.ckpt";
    if (log_path.empty())        log_path        = "vbz_throughput.log";

    std::ofstream log(log_path, std::ios::app);
    if (!log.is_open()) {
        std::cerr << "ERROR: cannot open log file: " << log_path << "\n";
        return 1;
    }
    // TSV header (only write if file is new / empty)
    {
        std::ifstream check_empty(log_path);
        check_empty.seekg(0, std::ios::end);
        if (check_empty.tellg() == 0) {
            log << "timestamp\tphase\tfiles_done\ttotal_files"
                << "\treads\tbytes_in\tbytes_out\tratio"
                << "\tcompress_s\tthroughput_MBs\tnote\n";
        }
    }

    auto logline = [&](const std::string& phase,
                       std::size_t files_done, std::size_t total_files,
                       uint64_t reads, uint64_t bytes_in, uint64_t bytes_out,
                       double compress_ns, const std::string& note = "")
    {
        double ratio = bytes_in > 0 ? double(bytes_out) / double(bytes_in) : 0.0;
        log << now_iso() << "\t"
            << phase << "\t"
            << files_done << "\t" << total_files << "\t"
            << reads << "\t" << bytes_in << "\t" << bytes_out << "\t"
            << std::fixed << std::setprecision(4) << ratio << "\t"
            << std::setprecision(3) << compress_ns / 1e9 << "\t"
            << std::setprecision(1) << to_mbs(bytes_in, compress_ns) << "\t"
            << note << "\n";
        log.flush();
    };

    // ---- Discover files ----------------------------------------------------
    std::cerr << "[" << now_iso() << "] Discovering fast5 files...\n";
    auto const files = discover_files(fast5_dir, fast5_list);
    if (files.empty()) {
        std::cerr << "ERROR: no .fast5 files found\n";
        return 1;
    }
    std::cerr << "[" << now_iso() << "] Found " << files.size() << " fast5 files\n";

    // ---- Dry run: sample a few files, then extrapolate ---------------------
    if (dry_run) {
        std::size_t sample_n = std::min(dry_run_files, files.size());
        std::cerr << "[" << now_iso() << "] Dry run: sampling " << sample_n
                  << " of " << files.size() << " files...\n";

        CompressionOptions opts{true, sizeof(int16_t), 1, VBZ_DEFAULT_VERSION};

        std::vector<std::vector<int16_t>> sample_signals;
        uint64_t sample_bytes = 0;
        for (std::size_t fi = 0; fi < sample_n; ++fi) {
            std::size_t before = sample_signals.size();
            load_fast5(files[fi], sample_signals);
            for (std::size_t r = before; r < sample_signals.size(); ++r)
                sample_bytes += sample_signals[r].size() * sizeof(int16_t);
        }

        if (sample_signals.empty()) {
            std::cerr << "ERROR: no signal data found in sample files\n";
            return 1;
        }

        std::size_t max_samples = 0;
        for (auto const& s : sample_signals)
            max_samples = std::max(max_samples, s.size());
        std::vector<char> dest(
            vbz_max_compressed_size(vbz_size_t(max_samples * sizeof(int16_t)), &opts));

        auto t0 = Clock::now();
        uint64_t compressed_bytes = 0;
        for (auto const& signal : sample_signals) {
            dest.resize(dest.capacity());
            auto csz = vbz_compress(signal.data(),
                                    vbz_size_t(signal.size() * sizeof(int16_t)),
                                    dest.data(), vbz_size_t(dest.size()), &opts);
            if (!vbz_is_error(csz)) compressed_bytes += csz;
        }
        double elapsed_ns = std::chrono::duration<double, std::nano>(
            Clock::now() - t0).count();

        double throughput_mbs  = to_mbs(sample_bytes, elapsed_ns);
        double bytes_per_file  = double(sample_bytes) / double(sample_n);
        double est_total_bytes = bytes_per_file * double(files.size());
        double est_total_s     = (throughput_mbs > 0.0)
                                 ? (est_total_bytes / 1e6) / throughput_mbs
                                 : 0.0;
        double est_load_s      = est_total_bytes / (1e9); // ~1 GB/s disk read
        double ratio           = sample_bytes > 0
                                 ? double(compressed_bytes) / double(sample_bytes)
                                 : 0.0;

        std::cerr << "\n=== Dry run results (sample: " << sample_n << " files, "
                  << sample_signals.size() << " reads, "
                  << std::fixed << std::setprecision(2)
                  << double(sample_bytes) / 1e6 << " MB) ===\n"
                  << "  Compression throughput : " << std::setprecision(1)
                                                   << throughput_mbs << " MB/s\n"
                  << "  Compression ratio      : " << std::setprecision(4) << ratio << "\n"
                  << "  Avg reads per file     : " << std::setprecision(0)
                                                   << double(sample_signals.size()) / double(sample_n) << "\n"
                  << "  Avg MB per file        : " << std::setprecision(1)
                                                   << bytes_per_file / 1e6 << "\n"
                  << "\n--- Full run estimates (" << files.size() << " files) ---\n"
                  << "  Total data (uncompressed) : " << std::setprecision(1)
                                                      << est_total_bytes / 1e9 << " GB\n"
                  << "  Estimated load time       : " << std::setprecision(1)
                                                      << est_load_s << " s  (assumes ~1 GB/s disk)\n"
                  << "  Estimated compress time   : " << est_total_s << " s\n"
                  << "  Estimated total time      : " << (est_load_s + est_total_s) << " s\n"
                  << "  Peak RAM needed           : ~" << std::setprecision(1)
                                                       << est_total_bytes / 1e9 << " GB\n";
        return 0;
    }

    // ---- Resume from checkpoint --------------------------------------------
    Checkpoint ckpt = load_checkpoint(checkpoint_path);
    if (ckpt.files_loaded > 0) {
        std::cerr << "[" << now_iso() << "] Resuming: " << ckpt.files_loaded
                  << "/" << files.size() << " files already loaded\n";
    }

    // ---- Phase 1: Load all signals into RAM --------------------------------
    std::cerr << "[" << now_iso() << "] Phase 1: loading signals into RAM...\n";

    std::vector<std::vector<int16_t>> all_signals;
    uint64_t load_reads_total = 0;
    uint64_t load_bytes_total = 0;

    auto load_start = Clock::now();
    auto last_log   = load_start;

    for (std::size_t fi = ckpt.files_loaded; fi < files.size(); ++fi) {
        if (g_interrupted) {
            std::cerr << "[" << now_iso() << "] Interrupted during load at file "
                      << fi << "/" << files.size() << ". Saving checkpoint.\n";
            save_checkpoint(checkpoint_path, ckpt);
            logline("load_interrupted", fi, files.size(),
                    load_reads_total, load_bytes_total, 0, 0.0, "interrupted");
            return 1;
        }

        std::size_t reads_before = all_signals.size();
        std::size_t n = load_fast5(files[fi], all_signals);
        if (n == 0) {
            std::cerr << "WARNING: no signal data in " << files[fi] << "\n";
        }

        for (std::size_t r = reads_before; r < all_signals.size(); ++r) {
            load_bytes_total += all_signals[r].size() * sizeof(int16_t);
        }
        load_reads_total += n;
        ckpt.files_loaded = fi + 1;

        // Periodic checkpoint
        if ((fi + 1) % checkpoint_interval == 0) {
            save_checkpoint(checkpoint_path, ckpt);
        }

        // Periodic progress log
        auto now = Clock::now();
        double elapsed_s = std::chrono::duration<double>(now - last_log).count();
        if (elapsed_s >= log_interval_s) {
            double total_elapsed = std::chrono::duration<double>(now - load_start).count();
            double rate_mbs = double(load_bytes_total) / 1e6 / total_elapsed;
            std::cerr << "[" << now_iso() << "] Load: "
                      << (fi + 1) << "/" << files.size() << " files, "
                      << load_reads_total << " reads, "
                      << std::fixed << std::setprecision(1)
                      << double(load_bytes_total) / 1e9 << " GB @ "
                      << rate_mbs << " MB/s\n";
            logline("loading", fi + 1, files.size(),
                    load_reads_total, load_bytes_total, 0, 0.0);
            last_log = now;
        }
    }

    save_checkpoint(checkpoint_path, ckpt);
    {
        double load_s = std::chrono::duration<double>(Clock::now() - load_start).count();
        std::cerr << "[" << now_iso() << "] Load complete: "
                  << all_signals.size() << " reads, "
                  << std::fixed << std::setprecision(2)
                  << double(load_bytes_total) / 1e9 << " GB in "
                  << std::setprecision(1) << load_s << "s\n";
        logline("load_complete", files.size(), files.size(),
                load_reads_total, load_bytes_total, 0, 0.0);
    }

    if (all_signals.empty()) {
        std::cerr << "ERROR: no signal data loaded\n";
        return 1;
    }

    // ---- Phase 2: Timed compression pass -----------------------------------
    std::cerr << "[" << now_iso() << "] Phase 2: compressing " << all_signals.size()
              << " reads (" << std::fixed << std::setprecision(2)
              << double(load_bytes_total) / 1e9 << " GB)...\n";

    CompressionOptions options{
        true,             // perform_delta_zig_zag
        sizeof(int16_t),  // integer_size
        1,                // zstd_compression_level
        VBZ_DEFAULT_VERSION
    };

    // Allocate output buffer large enough for any single read.
    std::size_t max_samples = 0;
    for (auto const& s : all_signals)
        max_samples = std::max(max_samples, s.size());
    std::vector<char> dest(
        vbz_max_compressed_size(vbz_size_t(max_samples * sizeof(int16_t)), &options));

    uint64_t compress_reads     = 0;
    uint64_t compress_bytes_in  = 0;
    uint64_t compress_bytes_out = 0;

    // Single outer timer — no per-call clock overhead in the hot path.
    auto compress_start = Clock::now();
    last_log = compress_start;

    for (std::size_t ri = 0; ri < all_signals.size(); ++ri) {
        if (g_interrupted) {
            // Snapshot elapsed before any further work.
            double elapsed_ns = std::chrono::duration<double, std::nano>(
                Clock::now() - compress_start).count();
            std::cerr << "[" << now_iso() << "] Interrupted during compression at read "
                      << ri << "/" << all_signals.size() << ". Writing partial log.\n";
            logline("compress_interrupted", files.size(), files.size(),
                    compress_reads, compress_bytes_in, compress_bytes_out,
                    elapsed_ns, "interrupted");
            break;
        }

        auto const& signal = all_signals[ri];
        auto const byte_count = vbz_size_t(signal.size() * sizeof(int16_t));
        dest.resize(dest.capacity());

        auto compressed_size = vbz_compress(
            signal.data(), byte_count,
            dest.data(), vbz_size_t(dest.size()),
            &options);

        if (vbz_is_error(compressed_size)) {
            std::cerr << "WARNING: vbz_compress error on read " << ri
                      << ": " << vbz_error_string(compressed_size) << "\n";
            continue;
        }

        compress_reads     += 1;
        compress_bytes_in  += byte_count;
        compress_bytes_out += compressed_size;

        // Progress logging: Clock::now() is called here only, outside the
        // compression timing window.
        auto now = Clock::now();
        double elapsed_s = std::chrono::duration<double>(now - last_log).count();
        if (elapsed_s >= log_interval_s) {
            double total_ns = std::chrono::duration<double, std::nano>(
                now - compress_start).count();
            double pct = 100.0 * double(ri + 1) / double(all_signals.size());
            std::cerr << "[" << now_iso() << "] Compress: "
                      << (ri + 1) << "/" << all_signals.size()
                      << " reads (" << std::fixed << std::setprecision(1) << pct << "%), "
                      << std::setprecision(2) << double(compress_bytes_in) / 1e9 << " GB in, "
                      << std::setprecision(1) << to_mbs(compress_bytes_in, total_ns) << " MB/s\n";
            logline("compressing", files.size(), files.size(),
                    compress_reads, compress_bytes_in, compress_bytes_out, total_ns);
            last_log = now;
        }
    }

    // ---- Final summary -----------------------------------------------------
    double total_ns = std::chrono::duration<double, std::nano>(
        Clock::now() - compress_start).count();
    double ratio = compress_bytes_in > 0
                   ? double(compress_bytes_out) / double(compress_bytes_in)
                   : 0.0;

    std::cerr << "\n[" << now_iso() << "] === Compression summary ===\n"
              << "  Reads compressed : " << compress_reads << "\n"
              << "  Input            : " << std::fixed << std::setprecision(3)
                                         << double(compress_bytes_in)  / 1e9 << " GB\n"
              << "  Output           : " << double(compress_bytes_out) / 1e9 << " GB\n"
              << "  Ratio            : " << std::setprecision(4) << ratio << "\n"
              << "  Elapsed          : " << std::setprecision(3) << total_ns / 1e9 << " s\n"
              << "  Throughput       : " << std::setprecision(1)
                                         << to_mbs(compress_bytes_in, total_ns) << " MB/s\n";

    logline("compress_complete", files.size(), files.size(),
            compress_reads, compress_bytes_in, compress_bytes_out,
            total_ns, "final");

    return 0;
}
