/* 直接测：反复 dlopen/dlclose NVIDIA Vulkan ICD 本身能做多少次。
 * 若在 ~26 次失败且报 static TLS，则证明 vkCreateInstance 的 -9 源自 loader dlopen ICD 失败。 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

int main(int argc, char **argv) {
    int rounds = (argc > 1) ? atoi(argv[1]) : 60;
    const char *lib = (argc > 2) ? argv[2] : "libGLX_nvidia.so.0";
    int keep = (argc > 3 && strcmp(argv[3], "--keep") == 0);
    void **hs = calloc(rounds, sizeof(void *));
    for (int i = 1; i <= rounds; i++) {
        void *h = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
        if (!h) { printf("DLCRASH round=%d err=%s\n", i, dlerror()); fflush(stdout); return 1; }
        printf("DLROUND %d ok\n", i); fflush(stdout);
        if (keep) hs[i - 1] = h; else dlclose(h);
    }
    printf("DLDONE rounds=%d keep=%d\n", rounds, keep);
    return 0;
}
