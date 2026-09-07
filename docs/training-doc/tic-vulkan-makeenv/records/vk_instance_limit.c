/* 独立最小复现：反复 vkCreateInstance，看单进程能建多少个 Vulkan instance。
 * --destroy 时每次建完立刻销毁，用于对照。不依赖 vulkan 头文件，手写最小结构体。 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <stdint.h>

typedef void *VkInstance;
typedef int32_t VkResult;

typedef struct {
    uint32_t sType; const void *pNext;
    const char *pApplicationName; uint32_t applicationVersion;
    const char *pEngineName; uint32_t engineVersion; uint32_t apiVersion;
} VkApplicationInfo;

typedef struct {
    uint32_t sType; const void *pNext; uint32_t flags;
    const VkApplicationInfo *pApplicationInfo;
    uint32_t enabledLayerCount; const char *const *ppEnabledLayerNames;
    uint32_t enabledExtensionCount; const char *const *ppEnabledExtensionNames;
} VkInstanceCreateInfo;

typedef VkResult (*PFN_vkCreateInstance)(const VkInstanceCreateInfo *, const void *, VkInstance *);
typedef void (*PFN_vkDestroyInstance)(VkInstance, const void *);
typedef VkResult (*PFN_vkEnumeratePhysicalDevices)(VkInstance, uint32_t *, void **);

static int count_fds(void) {
    char buf[256]; FILE *fp = popen("ls /proc/self/fd 2>/dev/null | wc -l", "r");
    int n = -1; if (fp) { if (fgets(buf, sizeof buf, fp)) n = atoi(buf); pclose(fp); }
    return n;
}

int main(int argc, char **argv) {
    int max_rounds = (argc > 1) ? atoi(argv[1]) : 64;
    int do_destroy = (argc > 2 && strcmp(argv[2], "--destroy") == 0);
    const char *libpath = getenv("VKLIB");
    if (!libpath) { fprintf(stderr, "需要设置 VKLIB 指向 libvulkan\n"); return 2; }

    void *h = dlopen(libpath, RTLD_NOW | RTLD_LOCAL);
    if (!h) { fprintf(stderr, "dlopen 失败: %s\n", dlerror()); return 2; }
    PFN_vkCreateInstance pCreate = (PFN_vkCreateInstance)dlsym(h, "vkCreateInstance");
    PFN_vkDestroyInstance pDestroy = (PFN_vkDestroyInstance)dlsym(h, "vkDestroyInstance");
    PFN_vkEnumeratePhysicalDevices pEnum = (PFN_vkEnumeratePhysicalDevices)dlsym(h, "vkEnumeratePhysicalDevices");
    if (!pCreate || !pDestroy) { fprintf(stderr, "dlsym 失败\n"); return 2; }

    VkApplicationInfo app; memset(&app, 0, sizeof app);
    app.sType = 0; app.pApplicationName = "probe"; app.pEngineName = "probe";
    app.apiVersion = (1u << 22) | (2u << 12);   /* VK_API_VERSION_1_2 */

    VkInstanceCreateInfo ci; memset(&ci, 0, sizeof ci);
    ci.sType = 1; ci.pApplicationInfo = &app;

    VkInstance *keep = calloc(max_rounds, sizeof(VkInstance));
    for (int i = 1; i <= max_rounds; i++) {
        VkInstance inst = NULL;
        VkResult r = pCreate(&ci, NULL, &inst);
        uint32_t ndev = 0;
        if (r == 0 && pEnum) pEnum(inst, &ndev, NULL);
        printf("VKROUND %d result=%d ndev=%u fd=%d\n", i, (int)r, ndev, count_fds());
        fflush(stdout);
        if (r != 0) { printf("VKCRASH at round=%d result=%d (VK_ERROR_INCOMPATIBLE_DRIVER=-9)\n", i, (int)r); fflush(stdout); return 1; }
        if (do_destroy) pDestroy(inst, NULL); else keep[i - 1] = inst;
    }
    printf("VKDONE rounds=%d destroy=%d\n", max_rounds, do_destroy);
    return 0;
}
