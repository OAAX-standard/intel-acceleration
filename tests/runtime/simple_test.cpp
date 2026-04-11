/**
 * Simple test program for Phase 2 OpenVINO native runtime
 *
 * This test verifies:
 * 1. Runtime initialization
 * 2. Runtime version and name
 * 3. send_input() returns -1 before a model is loaded
 * 4. runtime_error_message() returns the actual error after a failure
 * 5. Model loading (optional, requires a model path argument)
 * 6. Runtime destruction
 */

#include <cstring>
#include <iostream>

#include "../include/runtime_core.hpp"
#include "../include/tensors_struct.h"

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

    // Test 3: send_input before any model is loaded must return -1
    std::cout << "\n[Test 3] Calling send_input before model is loaded..." << std::endl;
    {
        tensors_struct dummy{};
        dummy.num_tensors = 0;
        int send_result = send_input(&dummy);
        if (send_result != -1) {
            std::cerr << "ERROR: send_input should return -1 before a model is loaded!" << std::endl;
            runtime_destruction();
            return 1;
        }
        std::cout << "✓ send_input correctly rejected (no model loaded)" << std::endl;
    }

    // Test 4: runtime_error_message returns the actual error after a failure
    std::cout << "\n[Test 4] Loading non-existent model (expect failure)..." << std::endl;
    result = runtime_model_loading("/nonexistent/path/model.xml");
    if (result != -1) {
        std::cerr << "ERROR: runtime_model_loading should have failed for a non-existent path!" << std::endl;
        runtime_destruction();
        return 1;
    }
    {
        const char* err = runtime_error_message();
        if (err == nullptr || strlen(err) == 0 || strcmp(err, "No error recorded.") == 0) {
            std::cerr << "ERROR: runtime_error_message() should return the actual error!" << std::endl;
            runtime_destruction();
            return 1;
        }
        std::cout << "✓ runtime_error_message set correctly: " << err << std::endl;
    }

    // Test 5: Model loading (if model path provided)
    if (argc > 1) {
        std::cout << "\n[Test 5] Loading model: " << argv[1] << std::endl;
        result = runtime_model_loading(argv[1]);
        if (result != 0) {
            std::cerr << "ERROR: Model loading failed!" << std::endl;
            std::cerr << runtime_error_message() << std::endl;
            runtime_destruction();
            return 1;
        }
        std::cout << "✓ Model loaded successfully" << std::endl;
    } else {
        std::cout << "\n[Test 5] Skipping model loading (no model provided)" << std::endl;
        std::cout << "Usage: " << argv[0] << " [model.xml]" << std::endl;
    }

    // Test 6: Runtime destruction
    std::cout << "\n[Test 6] Destroying runtime..." << std::endl;
    result = runtime_destruction();
    if (result != 0) {
        std::cerr << "ERROR: Runtime destruction failed!" << std::endl;
        return 1;
    }
    std::cout << "✓ Runtime destroyed successfully" << std::endl;

    std::cout << "\n=== All tests passed! ===" << std::endl;
    return 0;
}
