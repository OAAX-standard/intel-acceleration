#include "zip_utils.hpp"

#include <filesystem>
#include <random>
#include <sstream>
#include <string>

#include "miniz.h"

// ---------------------------------------------------------------------------
// Temp directory (cross-platform via std::filesystem)
// ---------------------------------------------------------------------------

static std::string create_temp_dir() {
  namespace fs = std::filesystem;
  auto base = fs::temp_directory_path();
  std::mt19937 gen(std::random_device{}());
  std::uniform_int_distribution<uint32_t> dist;

  for (int attempt = 0; attempt < 100; ++attempt) {
    std::ostringstream oss;
    oss << "oaax_model_" << std::hex << dist(gen);
    auto path = base / oss.str();
    std::error_code ec;
    if (fs::create_directory(path, ec)) return path.string();
  }
  return "";
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

std::string extract_zip_model(const std::string &zip_path,
                              std::string &out_temp_dir) {
  mz_zip_archive zip{};
  if (!mz_zip_reader_init_file(&zip, zip_path.c_str(), 0)) return "";

  mz_uint num_files = mz_zip_reader_get_num_files(&zip);
  if (num_files == 0) {
    mz_zip_reader_end(&zip);
    return "";
  }

  std::string temp_dir = create_temp_dir();
  if (temp_dir.empty()) {
    mz_zip_reader_end(&zip);
    return "";
  }
  out_temp_dir = temp_dir;

  std::string xml_path;

  for (mz_uint i = 0; i < num_files; ++i) {
    if (mz_zip_reader_is_file_a_directory(&zip, i)) continue;

    mz_zip_archive_file_stat stat;
    if (!mz_zip_reader_file_stat(&zip, i, &stat)) continue;

    std::string filename = stat.m_filename;
    std::string out_path =
        (std::filesystem::path(temp_dir) / filename).string();

    if (!mz_zip_reader_extract_to_file(&zip, i, out_path.c_str(), 0)) {
      std::error_code ec;
      std::filesystem::remove(out_path, ec);
      continue;
    }

    if (xml_path.empty() && filename.size() >= 4 &&
        filename.compare(filename.size() - 4, 4, ".xml") == 0) {
      xml_path = out_path;
    }
  }

  mz_zip_reader_end(&zip);
  return xml_path;
}

void cleanup_temp_dir(const std::string &temp_dir) {
  if (temp_dir.empty()) return;
  std::error_code ec;
  std::filesystem::remove_all(temp_dir, ec);
}
