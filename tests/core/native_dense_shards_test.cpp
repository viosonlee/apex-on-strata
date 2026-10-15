// NativeDense's split-header validation, including a repeated architecture name and
// an MTP marker in shard 2. Only F32 norm tensors are present: valid headers reach
// the no-matrix error before any CUDA allocation or launch. No GPU or model needed.
#include "gguf_fixture.hpp"
#include "strata/core/native_dense.hpp"
#include "strata/core/weights.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-80s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++failures;
}
struct TempDir {
    fs::path path = fs::temp_directory_path() /
        ("strata-dense-shards-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    TempDir() { fs::create_directories(path); }
    ~TempDir() { std::error_code ignored; fs::remove_all(path, ignored); }
};

std::vector<fixture::Kv> geometry(bool mtp) {
    return {fixture::str("general.architecture", "qwen35moe"), fixture::u32("qwen35moe.block_count", mtp ? 41 : 40),
            fixture::u32("qwen35moe.embedding_length", 2048), fixture::u32("qwen35moe.expert_count", 256),
            fixture::u32("qwen35moe.expert_used_count", 8), fixture::u32("qwen35moe.attention.head_count", 16),
            fixture::u32("qwen35moe.attention.head_count_kv", 2)};
}

std::vector<std::string> write_pair(const fs::path& dir, bool mtp, const std::string& later_arch,
                                  bool marker, uint64_t number = 1, bool duplicate = false) {
    auto first = geometry(mtp);
    for (const auto& key : fixture::split_keys(0, 2, 2)) first.push_back(key);
    auto last = fixture::split_keys(number, 2, 2);
    if (!later_arch.empty()) last.push_back(fixture::str("general.architecture", later_arch));
    const auto a = (dir / "model-00001-of-00002.gguf").string();
    const auto b = (dir / "model-00002-of-00002.gguf").string();
    fixture::write(a, first, {{"output_norm.weight", {8}, 0, 1}});
    const std::string name = duplicate ? "output_norm.weight" : marker ? "blk.40.nextn.eh_proj.weight" : "blk.0.attn_norm.weight";
    fixture::write(b, last, {{name, {8}, 0, 2}});
    return {a, b};
}

std::string load_error(const std::vector<std::string>& paths) {
    strata::core::WeightTable table;
    strata::core::NativeDense dense;
    std::string err;
    if (dense.load(paths, table, err, false)) return "unexpectedly loaded an empty dense model";
    return err;
}
} // namespace

int main() {
    const std::string passed_headers = "native dense: no supported GDN/QSA matrices in supplied shards";
    for (const std::string arch : {std::string(), std::string("qwen35moe")}) {
        TempDir dir;
        const auto paths = write_pair(dir.path, false, arch, false);
        const auto err = load_error(paths);
        check(err == passed_headers, "later shard with " + (arch.empty() ? std::string("no architecture") : arch) + ": " + err);
    }
    {
        TempDir dir;
        const auto paths = write_pair(dir.path, true, "qwen35moe", true);
        const strata::GgufModel model(paths);
        check(strata::check_architecture(model).empty(), "MTP marker in shard 2 satisfies the 41-layer model guard");
        check(!strata::check_architecture(model.meta()).empty(), "the metadata shard alone cannot prove an MTP block exists");
        check(load_error(paths) == passed_headers, "native dense validates the joint MTP model, not shard 1 alone");
    }
    {
        TempDir dir;
        check(load_error(write_pair(dir.path, true, "qwen35moe", false)).find("MTP") != std::string::npos,
              "41 layers without a real MTP marker refused");
    }
    {
        TempDir dir;
        check(load_error(write_pair(dir.path, false, "qwen35moe", true)).find("MTP") != std::string::npos,
              "40 layers with an MTP marker in a later shard refused");
    }
    {
        TempDir dir;
        const auto err = load_error(write_pair(dir.path, false, "qwen3", false));
        check(err.find("architecture-validated") != std::string::npos, "later shard of another architecture refused: " + err);
    }
    {
        TempDir dir;
        const auto err = load_error(write_pair(dir.path, false, "qwen35moe", false, 0));
        check(err.find("split.no") != std::string::npos, "duplicate shard number refused: " + err);
    }
    {
        TempDir dir;
        const auto err = load_error(write_pair(dir.path, false, "qwen35moe", false, 1, true));
        check(err.find("in two shards") != std::string::npos, "duplicate tensor refused: " + err);
    }
    std::printf("native_dense_shards_test: %d failure(s)\n", failures);
    return failures ? 1 : 0;
}
