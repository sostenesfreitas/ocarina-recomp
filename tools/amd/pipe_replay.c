// pipe_replay: builds one graphics pipeline dumped by the runtime (OOT_PIPE_DUMP=<hex hash of the pipeline note>, which
// writes pipe-dump-<hex>/ in the working folder: runtime/vk_pipeline.c)
// outside the game, on a device made like runtime/vk_device.c makes it, and times vkCreateGraphicsPipelines.
//
//   pipe_replay <pipe dir> [options]
//     --timeout S        give up after S seconds (default 120); the process exits with code 3 and "TIMEOUT"
//     --gpu N            physical device index (default: the first discrete GPU)
//     --opt / --noopt    clear / set VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT (default: as dumped)
//     --no-sg-pin        drop every stage's requiredSubgroupSize;  --sg N  pin N on the stages that had a pin
//     --stage I FILE     use FILE as the SPIR-V of stage I
//     --drop I           leave stage I out (a vertex-only pipeline when I is the fragment stage)
//     --gpl              enable VK_EXT_graphics_pipeline_library and build the pipeline as libraries instead
//                        (--gpl-only I: build only the library holding stage I)
//     --salt             add a unique OpSourceExtension to each module, so the driver's own cache misses (cold times)
//     --quiet            one line
//
// Build (from the repository root, with the bundled toolchain):
//   ferramentas\llvm\bin\clang -O2 -I ferramentas/deps/include tools/amd/pipe_replay.c -L ferramentas/deps/lib
//       -lvulkan-1 -o build/amd-debug/pipe_replay.exe
// The dumps hold the game's shaders: keep them on this machine.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#include <unistd.h>
#endif
#include <vulkan/vulkan.h>

static double now_s(void) {
#ifdef _WIN32
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
#endif
}

static void die(const char *m, int code) {
    fprintf(stderr, "pipe_replay: %s\n", m);
    exit(code);
}

// ---- pipeline.txt
typedef struct { uint32_t binding, type, count, stages; } Bind;
typedef struct { uint32_t flags, n; Bind *b; int bindless, present; } Set;
static struct {
    uint32_t flags;
    uint32_t n_stages, stage[4], req[4], stage_flags[4];
    char file[4][512];
    int vi;                       // a vertex input state was given
    uint32_t n_vb, n_va;
    VkVertexInputBindingDescription vb[32];
    VkVertexInputAttributeDescription va[32];
    int ia;
    uint32_t topo, restart;
    int vp, neg_one;
    uint32_t vp_n, sc_n;
    int rs;
    uint32_t clamp, discard, poly, cull, front, bias;
    float line_width;
    int provoking;
    int ms;
    uint32_t samples, sshade, a2c, a2one;
    float min_shade;
    int ds;
    uint32_t dtest, dwrite, dcmp, dbounds, stest;
    float dmin, dmax;
    int cb;
    uint32_t logic_en, logic_op, n_att;
    VkPipelineColorBlendAttachmentState att[8];
    int dyn;
    uint32_t n_dyn;
    VkDynamicState dyns[64];
    int rendering;
    uint32_t view_mask, depth_fmt, stencil_fmt, n_color, color[8];
    uint32_t layout_flags, n_sets;
    Set set[4];
} D;

static char *read_file(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *p = malloc((size_t)len + 1);
    if (!p || fread(p, 1, (size_t)len, f) != (size_t)len) die("cannot read a file", 2);
    fclose(f);
    p[len] = 0;
    if (n) *n = (size_t)len;
    return p;
}

static void parse(const char *dir) {
    char path[1024];
    snprintf(path, sizeof path, "%s/pipeline.txt", dir);
    char *txt = read_file(path, NULL);
    if (!txt) die("no pipeline.txt in that folder", 2);
    int cur_set = -1;
    for (char *line = strtok(txt, "\n"); line; line = strtok(NULL, "\n")) {
        char w[64];
        int k = 0;
        if (line[0] == '#' || sscanf(line, "%63s%n", w, &k) != 1) continue;
        const char *a = line + k;
        unsigned u[16];
        if (!strcmp(w, "flags")) sscanf(a, "%x", &D.flags);
        else if (!strcmp(w, "stage") && D.n_stages < 4) {
            unsigned st, req, fl;
            char file[512];
            if (sscanf(a, "%x %511s %u %x", &st, file, &req, &fl) == 4) {
                D.stage[D.n_stages] = st, D.req[D.n_stages] = req, D.stage_flags[D.n_stages] = fl;
                snprintf(D.file[D.n_stages], sizeof D.file[0], "%s/%s", dir, file);
                D.n_stages++;
            }
        } else if (!strcmp(w, "vertex_input")) D.vi = 1;
        else if (!strcmp(w, "vi_binding") && D.n_vb < 32 && sscanf(a, "%u %u %u", &u[0], &u[1], &u[2]) == 3)
            D.vb[D.n_vb++] = (VkVertexInputBindingDescription){u[0], u[1], (VkVertexInputRate)u[2]};
        else if (!strcmp(w, "vi_attr") && D.n_va < 32 && sscanf(a, "%u %u %u %u", &u[0], &u[1], &u[2], &u[3]) == 4)
            D.va[D.n_va++] = (VkVertexInputAttributeDescription){u[0], u[1], (VkFormat)u[2], u[3]};
        else if (!strcmp(w, "input_assembly") && sscanf(a, "%u %u", &D.topo, &D.restart) == 2) D.ia = 1;
        else if (!strcmp(w, "viewport") && sscanf(a, "%u %u %d", &D.vp_n, &D.sc_n, &D.neg_one) == 3) D.vp = 1;
        else if (!strcmp(w, "raster") && sscanf(a, "%u %u %u %u %u %u %f %d", &D.clamp, &D.discard, &D.poly, &D.cull, &D.front,
                                                &D.bias, &D.line_width, &D.provoking) == 8) D.rs = 1;
        else if (!strcmp(w, "multisample") && sscanf(a, "%u %u %f %u %u", &D.samples, &D.sshade, &D.min_shade, &D.a2c, &D.a2one) == 5) D.ms = 1;
        else if (!strcmp(w, "depth_stencil") && sscanf(a, "%u %u %u %u %u %f %f", &D.dtest, &D.dwrite, &D.dcmp, &D.dbounds, &D.stest,
                                                       &D.dmin, &D.dmax) == 7) D.ds = 1;
        else if (!strcmp(w, "blend") && sscanf(a, "%u %u %u", &D.logic_en, &D.logic_op, &D.n_att) == 3) D.cb = 1;
        else if (!strcmp(w, "blend_att")) {
            static uint32_t na;
            if (na < 8 && sscanf(a, "%u %u %u %u %u %u %u %x", &u[0], &u[1], &u[2], &u[3], &u[4], &u[5], &u[6], &u[7]) == 8)
                D.att[na++] = (VkPipelineColorBlendAttachmentState){u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7]};
        } else if (!strcmp(w, "dynamic")) {
            int off = 0, m = 0;
            unsigned n = 0, v;
            sscanf(a, "%u%n", &n, &off);
            a += off;
            while (m < 64 && sscanf(a, "%u%n", &v, &off) == 1) D.dyns[m++] = (VkDynamicState)v, a += off;
            D.n_dyn = (uint32_t)m, D.dyn = 1;
        } else if (!strcmp(w, "rendering")) {
            int off = 0;
            if (sscanf(a, "%u %u %u %u%n", &D.view_mask, &D.depth_fmt, &D.stencil_fmt, &D.n_color, &off) == 4) {
                a += off;
                for (uint32_t i = 0; i < D.n_color && i < 8; i++) {
                    int o2 = 0;
                    if (sscanf(a, "%u%n", &D.color[i], &o2) == 1) a += o2;
                }
                D.rendering = 1;
            }
        } else if (!strcmp(w, "layout")) sscanf(a, "%x %u", &D.layout_flags, &D.n_sets);
        else if (!strcmp(w, "set")) {
            unsigned idx, fl, n;
            char rest[32] = "";
            if (sscanf(a, "%u %31s", &idx, rest) == 2 && idx < 4 && !strcmp(rest, "bindless")) {
                D.set[idx].bindless = D.set[idx].present = 1;
                cur_set = -1;
            } else if (sscanf(a, "%u %x %u", &idx, &fl, &n) == 3 && idx < 4) {
                D.set[idx].flags = fl, D.set[idx].present = 1;
                D.set[idx].b = calloc(n ? n : 1, sizeof(Bind));
                cur_set = (int)idx;
            }
        } else if (!strcmp(w, "binding") && cur_set >= 0 && sscanf(a, "%u %u %u %x", &u[0], &u[1], &u[2], &u[3]) == 4) {
            Set *s = &D.set[cur_set];
            s->b[s->n++] = (Bind){u[0], u[1], u[2], u[3]};
        }
    }
    free(txt);
    if (!D.n_stages) die("pipeline.txt has no stages", 2);
}

// ---- device, as runtime/vk_device.c create_device makes it
static VkInstance inst;
static VkPhysicalDevice phys;
static VkDevice dev;

static int has_ext(const VkExtensionProperties *e, uint32_t n, const char *name) {
    for (uint32_t i = 0; i < n; i++)
        if (!strcmp(e[i].extensionName, name)) return 1;
    return 0;
}

typedef struct {
    VkPhysicalDeviceFeatures2 f2;
    VkPhysicalDeviceVulkan11Features f11;
    VkPhysicalDeviceVulkan12Features f12;
    VkPhysicalDeviceVulkan13Features f13;
    VkPhysicalDeviceDepthClipControlFeaturesEXT clip;
    VkPhysicalDeviceCustomBorderColorFeaturesEXT cbc;
    VkPhysicalDeviceExtendedDynamicStateFeaturesEXT eds;
    VkPhysicalDeviceExtendedDynamicState2FeaturesEXT eds2;
    VkPhysicalDeviceExtendedDynamicState3FeaturesEXT eds3;
    VkPhysicalDeviceVertexInputDynamicStateFeaturesEXT vids;
    VkPhysicalDeviceConditionalRenderingFeaturesEXT cond;
    VkPhysicalDeviceTransformFeedbackFeaturesEXT xfb;
    VkPhysicalDeviceProvokingVertexFeaturesEXT pv;
    VkPhysicalDeviceRobustness2FeaturesEXT rob2;
    VkPhysicalDeviceFaultFeaturesEXT fault;
    VkPhysicalDeviceMeshShaderFeaturesEXT mesh;
    VkPhysicalDeviceFragmentShaderInterlockFeaturesEXT fsi;
    VkPhysicalDeviceGraphicsPipelineLibraryFeaturesEXT gpl;
    VkPhysicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT fbl;
    VkPhysicalDeviceDynamicRenderingLocalReadFeaturesKHR lr;
} Chain;

static void make_device(int gpu, int want_gpl, int quiet) {
    const char *iext[] = {VK_KHR_SURFACE_EXTENSION_NAME,
#ifdef _WIN32
                          "VK_KHR_win32_surface"
#endif
    };
    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "TotK Recomp",
                             .pEngineName = "TotK Recomp", .apiVersion = VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app,
                                .enabledExtensionCount = sizeof iext / sizeof *iext, .ppEnabledExtensionNames = iext};
    if (vkCreateInstance(&ici, NULL, &inst) != VK_SUCCESS) die("vkCreateInstance failed", 2);
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, NULL);
    VkPhysicalDevice pd[16];
    if (n > 16) n = 16;
    vkEnumeratePhysicalDevices(inst, &n, pd);
    int pick = gpu;
    for (uint32_t i = 0; pick < 0 && i < n; i++) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(pd[i], &p);
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) pick = (int)i;
    }
    if (pick < 0) pick = 0;
    if ((uint32_t)pick >= n) die("no such gpu", 2);
    phys = pd[pick];
    VkPhysicalDeviceDriverProperties drv = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    VkPhysicalDeviceProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &drv};
    vkGetPhysicalDeviceProperties2(phys, &p2);
    if (!quiet) printf("gpu %d: %s, driver %s \"%s\"\n", pick, p2.properties.deviceName, drv.driverName, drv.driverInfo);

    uint32_t ne = 0;
    vkEnumerateDeviceExtensionProperties(phys, NULL, &ne, NULL);
    VkExtensionProperties *ep = calloc(ne ? ne : 1, sizeof *ep);
    vkEnumerateDeviceExtensionProperties(phys, NULL, &ne, ep);
    const char *ext[40];
    uint32_t n_ext = 0;
    ext[n_ext++] = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
    ext[n_ext++] = VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME;
    ext[n_ext++] = VK_EXT_DEPTH_CLIP_CONTROL_EXTENSION_NAME;
    ext[n_ext++] = VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME;
    Chain *c = calloc(1, sizeof *c);
    void **tail = &c->f2.pNext;
#define CHAIN(member, stype) do { c->member.sType = stype; *tail = &c->member; tail = &c->member.pNext; } while (0)
#define OPT(name, member, stype) do { if (has_ext(ep, ne, name)) { ext[n_ext++] = name; CHAIN(member, stype); } } while (0)
    c->f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    CHAIN(f11, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES);
    CHAIN(f12, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES);
    CHAIN(f13, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES);
    CHAIN(clip, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_CONTROL_FEATURES_EXT);
    OPT(VK_EXT_CUSTOM_BORDER_COLOR_EXTENSION_NAME, cbc, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT);
    OPT(VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME, eds, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT);
    OPT(VK_EXT_EXTENDED_DYNAMIC_STATE_2_EXTENSION_NAME, eds2, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_2_FEATURES_EXT);
    OPT(VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME, eds3, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT);
    OPT(VK_EXT_VERTEX_INPUT_DYNAMIC_STATE_EXTENSION_NAME, vids, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_INPUT_DYNAMIC_STATE_FEATURES_EXT);
    OPT(VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME, cond, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT);
    OPT(VK_EXT_TRANSFORM_FEEDBACK_EXTENSION_NAME, xfb, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_FEATURES_EXT);
    OPT(VK_EXT_PROVOKING_VERTEX_EXTENSION_NAME, pv, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROVOKING_VERTEX_FEATURES_EXT);
    OPT(VK_EXT_ROBUSTNESS_2_EXTENSION_NAME, rob2, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT);
    OPT(VK_EXT_MESH_SHADER_EXTENSION_NAME, mesh, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT);
    OPT(VK_EXT_DEVICE_FAULT_EXTENSION_NAME, fault, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT);
    OPT(VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME, fsi, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_INTERLOCK_FEATURES_EXT);
    if (has_ext(ep, ne, VK_EXT_ATTACHMENT_FEEDBACK_LOOP_LAYOUT_EXTENSION_NAME) && has_ext(ep, ne, VK_KHR_DYNAMIC_RENDERING_LOCAL_READ_EXTENSION_NAME)) {
        ext[n_ext++] = VK_EXT_ATTACHMENT_FEEDBACK_LOOP_LAYOUT_EXTENSION_NAME;
        ext[n_ext++] = VK_KHR_DYNAMIC_RENDERING_LOCAL_READ_EXTENSION_NAME;
        CHAIN(fbl, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ATTACHMENT_FEEDBACK_LOOP_LAYOUT_FEATURES_EXT);
        CHAIN(lr, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_LOCAL_READ_FEATURES_KHR);
    }
    if (has_ext(ep, ne, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)) ext[n_ext++] = VK_EXT_MEMORY_BUDGET_EXTENSION_NAME;
    if (has_ext(ep, ne, VK_AMD_BUFFER_MARKER_EXTENSION_NAME)) ext[n_ext++] = VK_AMD_BUFFER_MARKER_EXTENSION_NAME;
    if (want_gpl) {   // the game turns it off on this driver (TOTK_GPL=0); --gpl only
        if (!has_ext(ep, ne, VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME)) die("no VK_EXT_graphics_pipeline_library", 2);
        ext[n_ext++] = VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME;
        ext[n_ext++] = VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME;
        CHAIN(gpl, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_FEATURES_EXT);
    }
    vkGetPhysicalDeviceFeatures2(phys, &c->f2);   // everything the driver has, enabled (robustBufferAccess included)
    c->mesh.multiviewMeshShader = c->mesh.primitiveFragmentShadingRateMeshShader = c->mesh.meshShaderQueries = 0;
    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, NULL);
    VkQueueFamilyProperties qp[16];
    if (nq > 16) nq = 16;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, qp);
    uint32_t fam = 0;
    while (fam < nq && !(qp[fam].queueFlags & VK_QUEUE_GRAPHICS_BIT)) fam++;
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = fam, .queueCount = 1,
                                  .pQueuePriorities = &prio};
    VkDeviceCreateInfo di = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &c->f2, .queueCreateInfoCount = 1,
                             .pQueueCreateInfos = &qi, .enabledExtensionCount = n_ext, .ppEnabledExtensionNames = ext};
    VkResult r = vkCreateDevice(phys, &di, NULL, &dev);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "vkCreateDevice: %d\n", (int)r);
        die("vkCreateDevice failed", 2);
    }
    free(ep);
    free(c);
}

// ---- layout
static VkDescriptorSetLayout bindless_layout(void) {
    enum { BL_TEX = 32768, BL_SAMP = 4096, BL_KINDS = 8, BL_SAMP_BINDING = 8 };   // vk_exec_draw.c vk_bindless_layout
    VkDescriptorSetLayoutBinding b[BL_KINDS + 1];
    VkDescriptorBindingFlags f[BL_KINDS + 1];
    for (uint32_t i = 0; i <= BL_KINDS; i++) {
        int samp = i == BL_SAMP_BINDING;
        b[i] = (VkDescriptorSetLayoutBinding){i, samp ? VK_DESCRIPTOR_TYPE_SAMPLER : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                                              samp ? BL_SAMP : BL_TEX, VK_SHADER_STAGE_ALL, NULL};
        f[i] = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
               VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT;
    }
    VkDescriptorSetLayoutBindingFlagsCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
                                                      .bindingCount = BL_KINDS + 1, .pBindingFlags = f};
    VkDescriptorSetLayoutCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .pNext = &fi,
                                          .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
                                          .bindingCount = BL_KINDS + 1, .pBindings = b};
    VkDescriptorSetLayout l;
    if (vkCreateDescriptorSetLayout(dev, &ci, NULL, &l) != VK_SUCCESS) die("bindless set layout failed", 2);
    return l;
}

static VkPipelineLayout make_layout(void) {
    VkDescriptorSetLayout sl[4];
    uint32_t n = D.n_sets ? D.n_sets : 1;
    for (uint32_t i = 0; i < n; i++) {
        Set *s = &D.set[i];
        if (s->bindless) { sl[i] = bindless_layout(); continue; }
        VkDescriptorSetLayoutBinding b[512];
        for (uint32_t j = 0; j < s->n && j < 512; j++)
            b[j] = (VkDescriptorSetLayoutBinding){s->b[j].binding, (VkDescriptorType)s->b[j].type, s->b[j].count,
                                                  s->b[j].stages, NULL};
        VkDescriptorSetLayoutCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .flags = s->flags,
                                              .bindingCount = s->n, .pBindings = b};
        if (vkCreateDescriptorSetLayout(dev, &ci, NULL, &sl[i]) != VK_SUCCESS) die("set layout failed", 2);
    }
    VkPipelineLayoutCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .flags = D.layout_flags,
                                     .setLayoutCount = n, .pSetLayouts = sl};
    VkPipelineLayout l;
    if (vkCreatePipelineLayout(dev, &pi, NULL, &l) != VK_SUCCESS) die("pipeline layout failed", 2);
    return l;
}

// ---- the build, on its own thread so a hang can be timed out
static VkGraphicsPipelineCreateInfo G;
static volatile int done;
static volatile VkResult result;
static double t_build;

#ifdef _WIN32
static DWORD WINAPI build_thread(LPVOID arg)
#else
static void *build_thread(void *arg)
#endif
{
    (void)arg;
    VkPipeline p = VK_NULL_HANDLE;
    double t0 = now_s();
    result = vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &G, NULL, &p);
    t_build = now_s() - t0;
    done = 1;
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) die("usage: pipe_replay <pipe dir> [--timeout S] [--opt|--noopt] [--no-sg-pin] [--sg N] [--stage I FILE] "
                      "[--drop I] [--gpu N] [--quiet]", 2);
    const char *dir = argv[1];
    double timeout = 120;
    int gpu = -1, opt = -1, no_pin = 0, sg = 0, drop = -1, quiet = 0, gpl = 0, salt = 0;
    char *over[4] = {0};
    parse(dir);
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--timeout") && i + 1 < argc) timeout = atof(argv[++i]);
        else if (!strcmp(argv[i], "--gpu") && i + 1 < argc) gpu = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--opt")) opt = 1;
        else if (!strcmp(argv[i], "--noopt")) opt = 0;
        else if (!strcmp(argv[i], "--no-sg-pin")) no_pin = 1;
        else if (!strcmp(argv[i], "--sg") && i + 1 < argc) sg = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--stage") && i + 2 < argc) { int s = atoi(argv[++i]); if (s >= 0 && s < 4) over[s] = argv[++i]; else i++; }
        else if (!strcmp(argv[i], "--drop") && i + 1 < argc) drop = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--quiet")) quiet = 1;
        else if (!strcmp(argv[i], "--gpl")) gpl = 1;
        else if (!strcmp(argv[i], "--salt")) salt = 1;
        else die("unknown option", 2);
    }
    if (gpl) die("--gpl is not implemented yet", 2);
    make_device(gpu, gpl, quiet);
    VkPipelineLayout layout = make_layout();

    VkPipelineShaderStageCreateInfo st[4];
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rq[4];
    uint32_t ns = 0;
    char desc[512];
    size_t dl = 0;
    desc[0] = 0;
    for (uint32_t i = 0; i < D.n_stages; i++) {
        if ((int)i == drop) continue;
        size_t bytes;
        const char *file = over[i] ? over[i] : D.file[i];
        uint32_t *code = (uint32_t *)read_file(file, &bytes);
        if (!code) die("cannot read a stage file", 2);
        if (salt) {   // OpSourceExtension "salt <time> <stage>" before the first debug or annotation instruction
            size_t nw = bytes / 4, at = 5;
            while (at < nw) {
                uint32_t op = code[at] & 0xffff, len = code[at] >> 16;
                if (!len || op == 3 || op == 4 || op == 5 || op == 6 || op == 7 || op == 71 || op == 72) break;
                at += len;
            }
            char str[64];
            snprintf(str, sizeof str, "salt %.6f %u", now_s(), i);
            uint32_t sw = (uint32_t)(strlen(str) / 4 + 1);
            uint32_t *c2 = calloc(nw + 1 + sw, 4);
            memcpy(c2, code, at * 4);
            c2[at] = (1 + sw) << 16 | 4;
            memcpy(c2 + at + 1, str, strlen(str));
            memcpy(c2 + at + 1 + sw, code + at, (nw - at) * 4);
            free(code);
            code = c2, bytes = (nw + 1 + sw) * 4;
        }
        VkShaderModuleCreateInfo mi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = bytes, .pCode = code};
        VkShaderModule m;
        if (vkCreateShaderModule(dev, &mi, NULL, &m) != VK_SUCCESS) die("vkCreateShaderModule failed", 2);
        st[ns] = (VkPipelineShaderStageCreateInfo){.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                   .flags = D.stage_flags[i], .stage = (VkShaderStageFlagBits)D.stage[i],
                                                   .module = m, .pName = "main"};
        uint32_t req = no_pin ? 0 : D.req[i] && sg ? (uint32_t)sg : D.req[i];
        if (req) {
            rq[ns] = (VkPipelineShaderStageRequiredSubgroupSizeCreateInfo){
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO, .requiredSubgroupSize = req};
            st[ns].pNext = &rq[ns];
        }
        dl += (size_t)snprintf(desc + dl, sizeof desc - dl, "%s%#x:%zuw%s%u", ns ? " " : "", D.stage[i], bytes / 4,
                               req ? "/sg" : "/sg-", req);
        ns++;
    }
    VkPipelineVertexInputStateCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                                               .vertexBindingDescriptionCount = D.n_vb, .pVertexBindingDescriptions = D.vb,
                                               .vertexAttributeDescriptionCount = D.n_va, .pVertexAttributeDescriptions = D.va};
    VkPipelineInputAssemblyStateCreateInfo ia = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                                 .topology = (VkPrimitiveTopology)D.topo, .primitiveRestartEnable = D.restart};
    VkPipelineViewportDepthClipControlCreateInfoEXT clip = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_DEPTH_CLIP_CONTROL_CREATE_INFO_EXT,
                                                            .negativeOneToOne = D.neg_one > 0};
    VkPipelineViewportStateCreateInfo vp = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                                            .pNext = D.neg_one >= 0 ? &clip : NULL, .viewportCount = D.vp_n, .scissorCount = D.sc_n};
    VkPipelineRasterizationProvokingVertexStateCreateInfoEXT pvs = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_PROVOKING_VERTEX_STATE_CREATE_INFO_EXT,
        .provokingVertexMode = (VkProvokingVertexModeEXT)(D.provoking > 0 ? D.provoking : 0)};
    VkPipelineRasterizationStateCreateInfo rs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                                                 .pNext = D.provoking >= 0 ? &pvs : NULL, .depthClampEnable = D.clamp,
                                                 .rasterizerDiscardEnable = D.discard, .polygonMode = (VkPolygonMode)D.poly,
                                                 .cullMode = D.cull, .frontFace = (VkFrontFace)D.front, .depthBiasEnable = D.bias,
                                                 .lineWidth = D.line_width};
    VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                               .rasterizationSamples = (VkSampleCountFlagBits)(D.samples ? D.samples : 1),
                                               .sampleShadingEnable = D.sshade, .minSampleShading = D.min_shade,
                                               .alphaToCoverageEnable = D.a2c, .alphaToOneEnable = D.a2one};
    VkPipelineDepthStencilStateCreateInfo ds = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                                                .depthTestEnable = D.dtest, .depthWriteEnable = D.dwrite,
                                                .depthCompareOp = (VkCompareOp)D.dcmp, .depthBoundsTestEnable = D.dbounds,
                                                .stencilTestEnable = D.stest, .minDepthBounds = D.dmin, .maxDepthBounds = D.dmax};
    VkPipelineColorBlendStateCreateInfo cb = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                                              .logicOpEnable = D.logic_en, .logicOp = (VkLogicOp)D.logic_op,
                                              .attachmentCount = D.n_att, .pAttachments = D.att};
    VkPipelineDynamicStateCreateInfo dy = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                                           .dynamicStateCount = D.n_dyn, .pDynamicStates = D.dyns};
    VkFormat colors[8];
    for (uint32_t i = 0; i < 8; i++) colors[i] = (VkFormat)D.color[i];
    VkPipelineRenderingCreateInfo ri = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, .viewMask = D.view_mask,
                                        .colorAttachmentCount = D.n_color, .pColorAttachmentFormats = colors,
                                        .depthAttachmentFormat = (VkFormat)D.depth_fmt, .stencilAttachmentFormat = (VkFormat)D.stencil_fmt};
    uint32_t flags = D.flags;
    if (opt == 1) flags &= ~(uint32_t)VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT;
    if (opt == 0) flags |= VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT;
    G = (VkGraphicsPipelineCreateInfo){.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = D.rendering ? &ri : NULL,
                                       .flags = flags, .stageCount = ns, .pStages = st,
                                       .pVertexInputState = D.vi ? &vi : NULL, .pInputAssemblyState = D.ia ? &ia : NULL,
                                       .pViewportState = D.vp ? &vp : NULL, .pRasterizationState = D.rs ? &rs : NULL,
                                       .pMultisampleState = D.ms ? &ms : NULL, .pDepthStencilState = D.ds ? &ds : NULL,
                                       .pColorBlendState = D.cb ? &cb : NULL, .pDynamicState = D.dyn ? &dy : NULL,
                                       .layout = layout};
    if (!quiet) printf("building: flags %#x, stages %s, timeout %.0f s\n", flags, desc, timeout);
    fflush(stdout);
    double t0 = now_s();
#ifdef _WIN32
    HANDLE th = CreateThread(NULL, 64u << 20, build_thread, NULL, 0, NULL);
    if (!th) die("CreateThread failed", 2);
    while (!done && now_s() - t0 < timeout) Sleep(20);
#else
    pthread_t th;
    pthread_create(&th, NULL, build_thread, NULL);
    while (!done && now_s() - t0 < timeout) usleep(20000);
#endif
    if (!done) {
        printf("%s: TIMEOUT after %.1f s (flags %#x, %s)\n", dir, now_s() - t0, flags, desc);
        fflush(stdout);
#ifdef _WIN32
        ExitProcess(3);
#else
        _exit(3);
#endif
    }
    printf("%s: %s (VkResult %d) in %.3f s (flags %#x, %s)\n", dir, result == VK_SUCCESS ? "built" : "FAILED", (int)result, t_build, flags, desc);
    fflush(stdout);
#ifdef _WIN32
    ExitProcess(result == VK_SUCCESS ? 0 : 1);
#else
    _exit(result == VK_SUCCESS ? 0 : 1);
#endif
}
