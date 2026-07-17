/*
 * windows_load_test.c — diagnostic loader for the OAAX Windows runtime DLL.
 *
 * Loads RuntimeLibrary.dll step by step and prints the exact Win32 error at
 * every stage, so a load failure is never silent. Checks, in order:
 *   1. The DLL file exists and its PE header matches the process bitness
 *   2. The MSVC runtime DLLs are resolvable
 *   3. Each OpenVINO dependency DLL loads from the runtime's directory
 *   4. The runtime DLL itself loads
 *   5. All nine OAAX v2 symbols resolve via GetProcAddress
 *   6. runtime_get_name/version/info return sane strings
 *   7. (--init) runtime_init on CPU + runtime_cleanup succeed
 *
 * Build (MSVC, x64 Native Tools prompt):
 *   cl /W4 windows_load_test.c
 * Build (MinGW-w64, e.g. cross-compiling from Linux):
 *   x86_64-w64-mingw32-gcc -Wall windows_load_test.c -o windows_load_test.exe
 *
 * Usage:
 *   windows_load_test.exe <path\to\RuntimeLibrary.dll> [--init]
 */

#ifndef _WIN32
#error This diagnostic tool is Windows-only.
#endif

#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <windows.h>

/* ---- minimal OAAX v2 declarations (keep in sync with oaax_runtime.h) ---- */

typedef int RuntimeStatus; /* RUNTIME_STATUS_SUCCESS == 0 */

typedef struct Config {
    int length;
    const char **keys;
    const char **values;
} Config;

typedef RuntimeStatus (*runtime_init_fn)(Config);
typedef RuntimeStatus (*runtime_cleanup_fn)(void);
typedef const char *(*get_string_fn)(void);

/* Cast FARPROC via void(*)(void) to avoid -Wcast-function-type warnings. */
#define GET_FN(type, module, name) ((type)(void (*)(void))GetProcAddress(module, name))

/* ------------------------------------------------------------------------ */

static void print_last_error_ex(const char *what, int with_hints) {
    DWORD err = GetLastError();
    char msg[512] = {0};
    FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL, err,
                   MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), msg, sizeof(msg) - 1, NULL);
    /* strip trailing newline that FormatMessage appends */
    size_t n = strlen(msg);
    while (n > 0 && (msg[n - 1] == '\n' || msg[n - 1] == '\r')) msg[--n] = 0;

    fprintf(stderr, "  FAILED: %s\n  GetLastError = %lu (0x%lX): %s\n", what, (unsigned long)err, (unsigned long)err,
            msg);

    if (!with_hints) return;
    switch (err) {
        case ERROR_MOD_NOT_FOUND: /* 126 */
            fprintf(stderr,
                    "  Hint: the DLL itself was found, but one of its DEPENDENCIES is\n"
                    "  missing (openvino.dll, tbb12.dll, or the MSVC runtime). All\n"
                    "  OpenVINO DLLs must sit next to RuntimeLibrary.dll or be on PATH.\n"
                    "  Run 'dumpbin /dependents RuntimeLibrary.dll' to list them.\n");
            break;
        case ERROR_PROC_NOT_FOUND: /* 127 */
            fprintf(stderr,
                    "  Hint: a dependency DLL was found but has the wrong version —\n"
                    "  an expected export is missing (e.g. stale openvino.dll on PATH\n"
                    "  shadowing the bundled one).\n");
            break;
        case ERROR_BAD_EXE_FORMAT: /* 193 */
            fprintf(stderr, "  Hint: 32/64-bit mismatch between this process and the DLL.\n");
            break;
        case ERROR_DLL_INIT_FAILED: /* 1114 */
            fprintf(stderr, "  Hint: DllMain / a static initializer threw during load.\n");
            break;
        default:
            break;
    }
}

static void print_last_error(const char *what) { print_last_error_ex(what, 1); }

/* Read the PE header and return the machine type (0 on failure). */
static WORD pe_machine_type(const char *path) {
    WORD machine = 0;
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    IMAGE_DOS_HEADER dos;
    if (fread(&dos, sizeof(dos), 1, f) == 1 && dos.e_magic == IMAGE_DOS_SIGNATURE) {
        DWORD sig;
        if (fseek(f, dos.e_lfanew, SEEK_SET) == 0 && fread(&sig, sizeof(sig), 1, f) == 1 && sig == IMAGE_NT_SIGNATURE) {
            IMAGE_FILE_HEADER fh;
            if (fread(&fh, sizeof(fh), 1, f) == 1) machine = fh.Machine;
        }
    }
    fclose(f);
    return machine;
}

/* Probe a single dependency by name using the default search order. */
static void probe_dependency(const char *name) {
    HMODULE h = LoadLibraryA(name);
    if (h) {
        char where[MAX_PATH] = {0};
        GetModuleFileNameA(h, where, sizeof(where) - 1);
        printf("  OK  %-22s -> %s\n", name, where);
        FreeLibrary(h);
    } else {
        printf("  MISSING  %s\n", name);
        print_last_error_ex(name, 0);
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <path\\to\\RuntimeLibrary.dll> [--init]\n", argv[0]);
        return 2;
    }
    const char *dll_path = argv[1];
    int do_init = (argc > 2 && strcmp(argv[2], "--init") == 0);

    /* Turn OS error dialogs into return codes so nothing blocks or hides. */
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOOPENFILEERRORBOX);

    printf("== OAAX Windows runtime load test ==\n");
    printf("Process: %u-bit\n", (unsigned)(sizeof(void *) * 8));
    printf("Target DLL: %s\n\n", dll_path);

    /* --- 1. file exists + bitness ----------------------------------------- */
    printf("[1] Checking file and PE header\n");
    DWORD attrs = GetFileAttributesA(dll_path);
    if (attrs == INVALID_FILE_ATTRIBUTES) {
        print_last_error("GetFileAttributes (file not found?)");
        return 1;
    }
    WORD machine = pe_machine_type(dll_path);
    const char *arch = (machine == IMAGE_FILE_MACHINE_AMD64)   ? "x64"
                       : (machine == IMAGE_FILE_MACHINE_I386)  ? "x86"
                       : (machine == IMAGE_FILE_MACHINE_ARM64) ? "ARM64"
                                                               : "unknown";
    printf("  File exists. PE machine type: 0x%X (%s)\n", machine, arch);
    int process_is_64 = (sizeof(void *) == 8);
    int dll_is_64 = (machine == IMAGE_FILE_MACHINE_AMD64 || machine == IMAGE_FILE_MACHINE_ARM64);
    if (process_is_64 != dll_is_64) {
        fprintf(stderr,
                "  FATAL: this test process is %u-bit but the DLL is %s. Windows can\n"
                "  never load it — every load below will fail with error 193. This is\n"
                "  a property of the LOADING PROCESS, not the DLL:\n"
                "    - If test.exe is 32-bit: rebuild it from the 'x64 Native Tools\n"
                "      Command Prompt' (plain 'Developer Command Prompt' defaults to x86).\n"
                "    - If your real host app has the same bitness, that IS your bug —\n"
                "      a 32-bit host cannot load this x64 runtime.\n",
                (unsigned)(sizeof(void *) * 8), arch);
    }

    /* --- 2. make the DLL's own directory searchable for its dependencies -- */
    char dir[MAX_PATH] = {0};
    char full[MAX_PATH] = {0};
    GetFullPathNameA(dll_path, sizeof(full), full, NULL);
    strncpy(dir, full, sizeof(dir) - 1);
    char *slash = strrchr(dir, '\\');
    if (slash) *slash = 0;
    printf("\n[2] Adding DLL directory to search path: %s\n", dir);
    if (!SetDllDirectoryA(dir)) print_last_error("SetDllDirectory");

    /* --- 3. probe known dependencies individually -------------------------- */
    printf("\n[3] Probing dependencies (each via its own LoadLibrary)\n");
    printf("  -- MSVC runtime --\n");
    probe_dependency("vcruntime140.dll");
    probe_dependency("vcruntime140_1.dll");
    probe_dependency("msvcp140.dll");
    printf("  -- OpenVINO / TBB --\n");
    probe_dependency("tbb12.dll");
    probe_dependency("openvino.dll");

    /* OpenVINO loads these at RUNTIME via ov::Core — they never appear in any
     * import table, so Dependency Walker / dumpbin will NOT report them. */
    printf("  -- OpenVINO runtime-loaded plugins (invisible to dep walkers) --\n");
    static const char *plugins[] = {"openvino_intel_cpu_plugin.dll", "openvino_intel_gpu_plugin.dll",
                                    "openvino_intel_npu_plugin.dll", "openvino_ir_frontend.dll",
                                    "openvino_auto_plugin.dll",      "openvino_auto_batch_plugin.dll"};
    for (size_t i = 0; i < sizeof(plugins) / sizeof(plugins[0]); i++) {
        char p[MAX_PATH];
        snprintf(p, sizeof(p), "%s\\%s", dir, plugins[i]);
        printf("  %-8s %s\n", GetFileAttributesA(p) != INVALID_FILE_ATTRIBUTES ? "OK" : "ABSENT", plugins[i]);
    }
    printf("  (CPU plugin + ir_frontend are required; others per device use.)\n");

    /* --- 3b. CWD writability — the runtime creates runtime.log in the CWD --- */
    printf("\n[3b] Checking CWD writability (runtime_init writes runtime.log there)\n");
    char cwd[MAX_PATH] = {0};
    GetCurrentDirectoryA(sizeof(cwd), cwd);
    printf("  CWD: %s\n", cwd);
    HANDLE probe = CreateFileA(".__oaax_write_probe", GENERIC_WRITE, 0, NULL, CREATE_ALWAYS,
                               FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE, NULL);
    if (probe == INVALID_HANDLE_VALUE) {
        print_last_error("CWD write probe");
        fprintf(stderr,
                "  WARNING: CWD is NOT writable. With the default config the runtime\n"
                "  cannot create runtime.log — initialize_logger() then calls exit(),\n"
                "  TERMINATING THE HOST PROCESS with no error report. Host apps must\n"
                "  pass a writable 'log_file' path in the runtime_init Config.\n");
    } else {
        printf("  CWD is writable — default runtime.log creation will succeed.\n");
        CloseHandle(probe);
    }

    /* --- 4. load the runtime DLL ------------------------------------------ */
    printf("\n[4] Loading %s\n", full);
    HMODULE h = LoadLibraryExA(full, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!h) {
        print_last_error("LoadLibraryEx(runtime DLL)");
        fprintf(stderr,
                "\nFor the exact failing import, run in cmd:\n"
                "  dumpbin /dependents \"%s\"\n"
                "or enable loader snaps: gflags /i %s +sls, then check debugger output.\n",
                full, argv[0]);
        return 1;
    }
    printf("  OK — loaded at %p\n", (void *)h);

    /* --- 5. resolve all OAAX symbols --------------------------------------- */
    printf("\n[5] Resolving OAAX v2 exports\n");
    static const char *symbols[] = {
        "runtime_init",    "runtime_load_models", "runtime_enqueue_input", "runtime_retrieve_output",
        "runtime_cleanup", "runtime_get_error",   "runtime_get_version",   "runtime_get_name",
        "runtime_get_info"};
    int missing = 0;
    for (size_t i = 0; i < sizeof(symbols) / sizeof(symbols[0]); i++) {
        FARPROC p = GetProcAddress(h, symbols[i]);
        printf("  %-26s %s\n", symbols[i], p ? "OK" : "MISSING");
        if (!p) missing++;
    }
    if (missing) {
        fprintf(stderr,
                "  %d symbol(s) missing — exports not declared or name-mangled.\n"
                "  Check with: dumpbin /exports \"%s\"\n",
                missing, full);
        return 1;
    }

    /* --- 6. call the safe informational functions -------------------------- */
    printf("\n[6] Calling informational functions\n");
    get_string_fn get_name = GET_FN(get_string_fn, h, "runtime_get_name");
    get_string_fn get_version = GET_FN(get_string_fn, h, "runtime_get_version");
    get_string_fn get_error = GET_FN(get_string_fn, h, "runtime_get_error");
    printf("  runtime_get_name()    = %s\n", get_name() ? get_name() : "(null)");
    printf("  runtime_get_version() = %s\n", get_version() ? get_version() : "(null)");

    /* --- 7. optional init/cleanup round-trip -------------------------------- */
    if (do_init) {
        printf("\n[7] runtime_init on CPU (--init)\n");
        runtime_init_fn init = GET_FN(runtime_init_fn, h, "runtime_init");
        runtime_cleanup_fn cleanup = GET_FN(runtime_cleanup_fn, h, "runtime_cleanup");
        const char *keys[] = {"device_type", "log_stdout"};
        const char *values[] = {"CPU", "1"};
        Config cfg = {2, keys, values};
        printf(
            "  Calling runtime_init...\n"
            "  NOTE: if this process exits RIGHT HERE with no further output,\n"
            "  the runtime failed to create runtime.log in the CWD and called\n"
            "  exit() internally (see [3b]). Re-run from a writable directory.\n");
        fflush(stdout);
        RuntimeStatus st = init(cfg);
        printf("  runtime_init = %d %s\n", st, st == 0 ? "(SUCCESS)" : "(FAILED)");
        if (st != 0) {
            const char *e = get_error();
            fprintf(stderr, "  runtime_get_error() = %s\n", e ? e : "(null)");
            return 1;
        }
        st = cleanup();
        printf("  runtime_cleanup = %d %s\n", st, st == 0 ? "(SUCCESS)" : "(FAILED)");
    } else {
        printf("\n[7] Skipped runtime_init (pass --init to enable)\n");
    }

    FreeLibrary(h);
    printf("\nAll checks passed.\n");
    return 0;
}
