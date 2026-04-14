#pragma once
#include <string>

/**
 * @brief Extract an OpenVINO IR model from a toolchain-produced zip archive.
 *
 * The archive must contain a `.xml` file and a co-located `.bin` file at its
 * root (no subdirectories).  The files are extracted to a newly-created
 * temporary directory and the path to the first `.xml` found is returned.
 *
 * @p out_temp_dir is set to the temp directory path on success; the caller
 * must later remove it via cleanup_temp_dir().
 *
 * @param zip_path     Path to the zip archive.
 * @param out_temp_dir Filled with the created temp directory path on success.
 * @return Path to the extracted `.xml` file, or empty string on failure.
 */
std::string extract_zip_model(const std::string &zip_path,
                              std::string &out_temp_dir);

/**
 * @brief Recursively remove the temp directory created by extract_zip_model().
 *
 * Safe to call with an empty string (no-op).
 */
void cleanup_temp_dir(const std::string &temp_dir);
