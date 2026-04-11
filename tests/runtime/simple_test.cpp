/**
 * Simple test program for Phase 2 OpenVINO native runtime
 *
 * This test verifies:
 * 1. Runtime initialization
 * 2. Model loading capability
 * 3. Runtime destruction
 */

#include <cstring>
#include <iostream>

#include "../include/runtime_core.hpp"

int main(int argc, char** argv) {
  std::cout << "=== OAAX OpenVINO Native Runtime Test ===" << std::endl;

  // Test 1: Runtime initialization
  std::cout << "\n[Test 1] Initializing runtime..." << std::endl;
  int result = runtime_initialization();
  if (result != 0) {
    std::cerr << "ERROR: Runtime initialization failed!" << std::endl;
    return 1;
  }
  std::cout << "✓ Runtime initialized successfully" << std::endl;

  // Test 2: Runtime version and name
  std::cout << "\n[Test 2] Checking runtime info..." << std::endl;
  const char* version = runtime_version();
  const char* name = runtime_name();
  std::cout << "Runtime Name: " << name << std::endl;
  std::cout << "Runtime Version: " << version << std::endl;

  if (strcmp(name, "OAAX Intel Runtime (OpenVINO Native)") != 0) {
    std::cerr << "ERROR: Unexpected runtime name!" << std::endl;
    return 1;
  }
  std::cout << "✓ Runtime info correct" << std::endl;

  // Test 3: Model loading (if model path provided)
  if (argc > 1) {
    std::cout << "\n[Test 3] Loading model: " << argv[1] << std::endl;
    result = runtime_model_loading(argv[1]);
    if (result != 0) {
      std::cerr << "ERROR: Model loading failed!" << std::endl;
      std::cerr << runtime_error_message() << std::endl;
      runtime_destruction();
      return 1;
    }
    std::cout << "✓ Model loaded successfully" << std::endl;
  } else {
    std::cout << "\n[Test 3] Skipping model loading (no model provided)" << std::endl;
    std::cout << "Usage: " << argv[0] << " [model.xml]" << std::endl;
  }

  // Test 4: Runtime destruction
  std::cout << "\n[Test 4] Destroying runtime..." << std::endl;
  result = runtime_destruction();
  if (result != 0) {
    std::cerr << "ERROR: Runtime destruction failed!" << std::endl;
    return 1;
  }
  std::cout << "✓ Runtime destroyed successfully" << std::endl;

  std::cout << "\n=== All tests passed! ===" << std::endl;
  return 0;
}
