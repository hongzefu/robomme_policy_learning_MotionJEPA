/* C 层实验 v2：instance + logical device 都不销毁，看单进程上限。
 * 用法: VKLIB=... VK_ICD_FILENAMES=... ./vk_device_limit <rounds> <phys_index> [--destroy] */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <stdint.h>

typedef void *VkInstance; typedef void *VkPhysicalDevice; typedef void *VkDevice;
typedef int32_t VkResult;

typedef struct { uint32_t sType; const void *pNext; const char *pApplicationName;
    uint32_t applicationVersion; const char *pEngineName; uint32_t engineVersion;
    uint32_t apiVersion; } VkApplicationInfo;
typedef struct { uint32_t sType; const void *pNext; uint32_t flags;
    const VkApplicationInfo *pApplicationInfo; uint32_t enabledLayerCount;
    const char *const *ppEnabledLayerNames; uint32_t enabledExtensionCount;
    const char *const *ppEnabledExtensionNames; } VkInstanceCreateInfo;
typedef struct { uint32_t sType; const void *pNext; uint32_t flags;
    uint32_t queueFamilyIndex; uint32_t queueCount; const float *pQueuePriorities; } VkDeviceQueueCreateInfo;
typedef struct { uint32_t sType; const void *pNext; uint32_t flags;
    uint32_t queueCreateInfoCount; const VkDeviceQueueCreateInfo *pQueueCreateInfos;
    uint32_t enabledLayerCount; const char *const *ppEnabledLayerNames;
    uint32_t enabledExtensionCount; const char *const *ppEnabledExtensionNames;
    const void *pEnabledFeatures; } VkDeviceCreateInfo;

typedef VkResult (*PFN_CI)(const VkInstanceCreateInfo *, const void *, VkInstance *);
typedef void (*PFN_DI)(VkInstance, const void *);
typedef VkResult (*PFN_EPD)(VkInstance, uint32_t *, VkPhysicalDevice *);
typedef VkResult (*PFN_CD)(VkPhysicalDevice, const VkDeviceCreateInfo *, const void *, VkDevice *);
typedef void (*PFN_DD)(VkDevice, const void *);

static int count_fds(void) {
    char buf[256]; FILE *fp = popen("ls /proc/self/fd 2>/dev/null | wc -l", "r");
    int n = -1; if (fp) { if (fgets(buf, sizeof buf, fp)) n = atoi(buf); pclose(fp); } return n;
}
static int count_nvidia_maps(void) {
    char buf[256]; FILE *fp = popen("grep -c '/dev/nvidia' /proc/self/maps 2>/dev/null", "r");
    int n = -1; if (fp) { if (fgets(buf, sizeof buf, fp)) n = atoi(buf); pclose(fp); } return n;
}

int main(int argc, char **argv) {
    int max_rounds = (argc > 1) ? atoi(argv[1]) : 64;
    int phys_idx = (argc > 2) ? atoi(argv[2]) : 0;
    int do_destroy = (argc > 3 && strcmp(argv[3], "--destroy") == 0);
    const char *libpath = getenv("VKLIB");
    if (!libpath) { fprintf(stderr, "需要 VKLIB\n"); return 2; }
    void *h = dlopen(libpath, RTLD_NOW | RTLD_LOCAL);
    if (!h) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 2; }
    PFN_CI pCI = (PFN_CI)dlsym(h, "vkCreateInstance");
    PFN_DI pDI = (PFN_DI)dlsym(h, "vkDestroyInstance");
    PFN_EPD pEPD = (PFN_EPD)dlsym(h, "vkEnumeratePhysicalDevices");
    PFN_CD pCD = (PFN_CD)dlsym(h, "vkCreateDevice");
    PFN_DD pDD = (PFN_DD)dlsym(h, "vkDestroyDevice");
    if (!pCI || !pEPD || !pCD) { fprintf(stderr, "dlsym 失败\n"); return 2; }

    VkApplicationInfo app; memset(&app, 0, sizeof app);
    app.sType = 0; app.pApplicationName = "probe"; app.pEngineName = "probe";
    app.apiVersion = (1u << 22) | (2u << 12);
    VkInstanceCreateInfo ici; memset(&ici, 0, sizeof ici);
    ici.sType = 1; ici.pApplicationInfo = &app;
    float prio = 1.0f;

    for (int i = 1; i <= max_rounds; i++) {
        VkInstance inst = NULL;
        VkResult r = pCI(&ici, NULL, &inst);
        if (r != 0) { printf("VKCRASH stage=instance round=%d result=%d\n", i, (int)r); fflush(stdout); return 1; }
        uint32_t n = 0; pEPD(inst, &n, NULL);
        VkPhysicalDevice *pds = calloc(n ? n : 1, sizeof(VkPhysicalDevice));
        pEPD(inst, &n, pds);
        int use = (phys_idx < (int)n) ? phys_idx : 0;
        VkDeviceQueueCreateInfo q; memset(&q, 0, sizeof q);
        q.sType = 2; q.queueFamilyIndex = 0; q.queueCount = 1; q.pQueuePriorities = &prio;
        VkDeviceCreateInfo dci; memset(&dci, 0, sizeof dci);
        dci.sType = 3; dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &q;
        VkDevice dev = NULL;
        VkResult rd = pCD(pds[use], &dci, NULL, &dev);
        printf("VKROUND %d ndev=%u phys=%d inst_ok=1 dev_result=%d fd=%d nvmaps=%d\n",
               i, n, use, (int)rd, count_fds(), count_nvidia_maps());
        fflush(stdout);
        if (rd != 0) { printf("VKCRASH stage=device round=%d result=%d\n", i, (int)rd); fflush(stdout); return 1; }
        if (do_destroy) { if (pDD) pDD(dev, NULL); if (pDI) pDI(inst, NULL); }
        free(pds);
    }
    printf("VKDONE rounds=%d destroy=%d\n", max_rounds, do_destroy);
    return 0;
}
