#include "zip_utils.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <compressapi.h>
#include <windows.h>
#pragma comment(lib, "Cabinet.lib")
#else
#include <zlib.h>
#endif

// ---------------------------------------------------------------------------
// Little-endian reads from a raw byte buffer
// ---------------------------------------------------------------------------

static uint16_t le16(const uint8_t *p) {
  return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static uint32_t le32(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
         ((uint32_t)p[3] << 24);
}

// ---------------------------------------------------------------------------
// End of Central Directory search
// ---------------------------------------------------------------------------

// Searches the last min(65557, file_size) bytes for the EOCD signature so it
// tolerates zip archives with a comment.
static bool find_eocd(FILE *f, uint32_t &cd_offset, uint16_t &cd_count) {
  if (fseek(f, 0, SEEK_END) != 0) return false;
  long file_size = ftell(f);
  if (file_size < 22) return false;

  long buf_size = std::min(file_size, 65557L);
  std::vector<uint8_t> buf(buf_size);
  if (fseek(f, file_size - buf_size, SEEK_SET) != 0) return false;
  if ((long)fread(buf.data(), 1, buf_size, f) != buf_size) return false;

  for (long i = buf_size - 22; i >= 0; --i) {
    if (buf[i] == 0x50 && buf[i + 1] == 0x4b && buf[i + 2] == 0x05 &&
        buf[i + 3] == 0x06) {
      cd_count = le16(buf.data() + i + 10);   // total central directory entries
      cd_offset = le32(buf.data() + i + 16);  // offset of central directory
      return true;
    }
  }
  return false;
}

// ---------------------------------------------------------------------------
// Platform-specific DEFLATE decompression (zip compression method 8)
// ---------------------------------------------------------------------------

#ifdef _WIN32

// Uses the Windows Compression API (Cabinet.dll, available since Windows 8).
// COMPRESS_ALGORITHM_DEFLATE is raw RFC 1951 DEFLATE — exactly what zip uses.
static bool inflate_entry(FILE *in, uint32_t comp_size, uint32_t uncomp_size,
                          FILE *out) {
  std::vector<uint8_t> compressed(comp_size);
  if (fread(compressed.data(), 1, comp_size, in) != comp_size) return false;

  DECOMPRESSOR_HANDLE handle = nullptr;
  if (!CreateDecompressor(COMPRESS_ALGORITHM_DEFLATE, nullptr, &handle))
    return false;

  std::vector<uint8_t> decompressed(uncomp_size);
  SIZE_T actual = uncomp_size;
  BOOL ok = Decompress(handle, compressed.data(), (SIZE_T)comp_size,
                       decompressed.data(), (SIZE_T)uncomp_size, &actual);
  CloseDecompressor(handle);

  if (!ok) return false;
  return fwrite(decompressed.data(), 1, actual, out) == actual;
}

#else

// Uses zlib inflate with windowBits=-15 (raw deflate, no zlib header/trailer).
static bool inflate_entry(FILE *in, uint32_t comp_size,
                          uint32_t /*uncomp_size*/, FILE *out) {
  const size_t CHUNK = 65536;
  std::vector<uint8_t> in_buf(CHUNK), out_buf(CHUNK);

  z_stream strm{};
  if (inflateInit2(&strm, -15) != Z_OK) return false;

  uint32_t remaining = comp_size;
  int ret = Z_OK;

  while (remaining > 0 && ret != Z_STREAM_END) {
    size_t to_read = std::min((size_t)remaining, CHUNK);
    size_t nread = fread(in_buf.data(), 1, to_read, in);
    if (nread == 0) break;
    remaining -= (uint32_t)nread;

    strm.next_in = in_buf.data();
    strm.avail_in = (uInt)nread;

    do {
      strm.next_out = out_buf.data();
      strm.avail_out = (uInt)CHUNK;
      ret = inflate(&strm, Z_NO_FLUSH);
      if (ret == Z_STREAM_ERROR || ret == Z_DATA_ERROR || ret == Z_MEM_ERROR) {
        inflateEnd(&strm);
        return false;
      }
      size_t produced = CHUNK - strm.avail_out;
      if (fwrite(out_buf.data(), 1, produced, out) != produced) {
        inflateEnd(&strm);
        return false;
      }
    } while (strm.avail_out == 0);
  }

  inflateEnd(&strm);
  return ret == Z_STREAM_END;
}

#endif  // _WIN32

// ---------------------------------------------------------------------------
// STORE entry (compression method 0 — verbatim copy)
// ---------------------------------------------------------------------------

static bool copy_entry(FILE *in, uint32_t comp_size, FILE *out) {
  std::vector<uint8_t> buf(65536);
  uint32_t remaining = comp_size;
  while (remaining > 0) {
    size_t chunk = std::min((size_t)remaining, buf.size());
    if (fread(buf.data(), 1, chunk, in) != chunk) return false;
    if (fwrite(buf.data(), 1, chunk, out) != chunk) return false;
    remaining -= (uint32_t)chunk;
  }
  return true;
}

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
  FILE *f = fopen(zip_path.c_str(), "rb");
  if (!f) return "";

  uint32_t cd_offset;
  uint16_t cd_count;
  if (!find_eocd(f, cd_offset, cd_count) || cd_count == 0) {
    fclose(f);
    return "";
  }

  std::string temp_dir = create_temp_dir();
  if (temp_dir.empty()) {
    fclose(f);
    return "";
  }
  out_temp_dir = temp_dir;

  std::string xml_path;

  if (fseek(f, (long)cd_offset, SEEK_SET) != 0) {
    fclose(f);
    return "";
  }

  for (uint16_t i = 0; i < cd_count; ++i) {
    // Central directory entry fixed header: 46 bytes.
    uint8_t cd[46];
    if (fread(cd, 1, 46, f) != 46) break;
    if (le32(cd) != 0x02014b50u) break;  // bad signature

    uint16_t compression = le16(cd + 10);
    uint32_t comp_size = le32(cd + 20);
    uint32_t uncomp_size = le32(cd + 24);
    uint16_t fname_len = le16(cd + 28);
    uint16_t extra_len = le16(cd + 30);
    uint16_t comment_len = le16(cd + 32);
    uint32_t local_offset = le32(cd + 42);

    std::string filename(fname_len, '\0');
    if (fread(filename.data(), 1, fname_len, f) != fname_len) break;
    if (fseek(f, extra_len + comment_len, SEEK_CUR) != 0) break;
    long cd_pos = ftell(f);

    // Skip directory entries.
    if (!filename.empty() && filename.back() == '/') continue;

    // Jump to the local file header to get the exact data offset.
    if (fseek(f, (long)local_offset, SEEK_SET) != 0) {
      fseek(f, cd_pos, SEEK_SET);
      continue;
    }

    uint8_t lh[30];
    if (fread(lh, 1, 30, f) != 30 || le32(lh) != 0x04034b50u) {
      fseek(f, cd_pos, SEEK_SET);
      continue;
    }
    uint16_t lh_fname_len = le16(lh + 26);
    uint16_t lh_extra_len = le16(lh + 28);
    if (fseek(f, lh_fname_len + lh_extra_len, SEEK_CUR) != 0) {
      fseek(f, cd_pos, SEEK_SET);
      continue;
    }

    // Build output path using filesystem to handle any OS separator.
    std::string out_path =
        (std::filesystem::path(temp_dir) / filename).string();
    FILE *out = fopen(out_path.c_str(), "wb");
    if (!out) {
      fseek(f, cd_pos, SEEK_SET);
      continue;
    }

    bool ok = false;
    if (compression == 0)
      ok = copy_entry(f, comp_size, out);
    else if (compression == 8)
      ok = inflate_entry(f, comp_size, uncomp_size, out);

    fclose(out);

    if (!ok) {
      std::error_code ec;
      std::filesystem::remove(out_path, ec);
    } else if (xml_path.empty() && fname_len >= 4 &&
               filename.compare(filename.size() - 4, 4, ".xml") == 0) {
      xml_path = out_path;
    }

    fseek(f, cd_pos, SEEK_SET);
  }

  fclose(f);
  return xml_path;
}

void cleanup_temp_dir(const std::string &temp_dir) {
  if (temp_dir.empty()) return;
  std::error_code ec;
  std::filesystem::remove_all(temp_dir, ec);
}
