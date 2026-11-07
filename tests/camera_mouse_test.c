// Exercise the production camera hooks with synthetic motion; no game files or window.
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "cfg.h"
#include "input.h"
#include "nvn_api.h"
#include "platform/platform.h"

void ov_camera_mouse(uint64_t node);
void ov_camera_aim_mouse(uint64_t node, uint64_t angles);
float ov_camera_axis_mouse(uint64_t node, int axis, float delta, float speed);
int ov_camera_tail_mouse_idle(uint64_t node, int idle);
GfxStats gfx_counters;
static int captured = 1, takes;
static int32_t raw_x, raw_y;
static double sensitivity = 1;
float input_mouse_sensitivity(void) { return (float)sensitivity; }
int ikb_captured(void) { return captured; }
void ikb_mouse_take(int32_t *x, int32_t *y) { *x = raw_x; *y = raw_y; raw_x = raw_y = 0; takes++; }
void plat_mutex_lock(PlatMutex *m) { (void)m; }
void plat_mutex_unlock(PlatMutex *m) { (void)m; }
static void near(float a, float b) { assert(fabsf(a - b) < .001f); }
_Alignas(8) static uint8_t free_node[0x250], aim[0x250], shared[0x800];
static float angles[3];
#define F(p, o) (*(float *)((p) + (o)))
#define U(p, o) (*(uint32_t *)((p) + (o)))
static void motion(int x, int y) { gfx_counters.frames++; raw_x = x; raw_y = y; }
static void reset(void) {
    memset(free_node, 0, sizeof free_node); memset(aim, 0, sizeof aim); memset(shared, 0, sizeof shared);
    *(uint8_t **)(aim + 0x20) = shared;
    F(aim, 0x224) = -35; F(aim, 0x228) = 40;
    angles[0] = 5; angles[1] = angles[2] = 0;
}
static void aim_update(void) { ov_camera_aim_mouse((uintptr_t)aim, (uintptr_t)angles); }
static void free_update(void) { ov_camera_mouse((uintptr_t)free_node); }
int main(void) {
    assert(!input_camera_mouse_active());
    reset(); motion(100, -50);
    free_update(); aim_update();
    assert(input_camera_mouse_active());
    near(F(free_node, 0x1a4), -6); near(F(free_node, 0x18c), -3);
    near(angles[2], -6); near(angles[1], -3); near(angles[0], 5);
    near(F(aim, 0x190), -6); near(F(aim, 0x18c), -3); near(F(aim, 0x194), -3);
    assert(takes == 1);
    aim_update(); free_update(); near(angles[2], -6); near(F(free_node, 0x1a4), -6); assert(takes == 1);

    // Reverse controller order during a transition, and preserve the stick/gyro's existing angle.
    motion(50, 25); angles[1] = 7; angles[2] = 20;
    F(aim, 0x194) = 4; F(aim, 0x18c) = 8; F(aim, 0x190) = 30;
    aim_update(); free_update();
    near(angles[1], 8.5); near(angles[2], 17); near(F(aim, 0x194), 5.5); near(F(aim, 0x18c), 9.5);
    near(F(aim, 0x190), 27); near(F(free_node, 0x1a4), -9); assert(takes == 2);
    motion(0, 0); aim_update(); near(angles[1], 8.5); near(angles[2], 17);

    // Respect aim parameters and the game's more restrictive submode, even with huge flicks.
    reset(); motion(100000, 100000); aim_update();
    near(angles[1], 40); near(F(aim, 0x194), 40); assert(fabsf(angles[2]) <= 180);
    reset(); U(aim, 0x23c) = 2; F(aim, 0x224) = -10; F(aim, 0x228) = 12;
    motion(-1000, 1000); aim_update(); near(angles[1], 5); near(angles[2], 12);
    near(F(aim, 0x190), 12); near(F(aim, 0x18c), 5);
    reset(); *(uint64_t *)(shared + 0x218) = UINT64_C(0x40000000);
    motion(100, 100); aim_update(); near(angles[1], 0); near(angles[2], 0);
    reset(); F(aim, 0x224) = NAN; motion(100, 100); aim_update(); near(angles[1], 0);

    // Gyro can take the output beyond the base limit. Horizontal mouse motion
    // must not snap that pitch; a saturated base retains the gyro offset.
    reset(); F(aim, 0x194) = F(aim, 0x18c) = 39; angles[1] = 42;
    motion(100, 0); aim_update(); near(angles[1], 42); near(F(aim, 0x194), 39);
    motion(0, 100); aim_update(); near(angles[1], 43); near(F(aim, 0x194), 40);
    motion(0, 100); aim_update(); near(angles[1], 43);

    // Releasing capture discards pending movement, including a sample already seen by the other camera.
    reset(); motion(100, 0); free_update(); captured = 0; raw_x = 200; aim_update();
    captured = 1; aim_update(); near(angles[2], 0); assert(raw_x == 0);
    sensitivity = 0;  motion(100, 100); aim_update(); near(angles[2], 0);
    sensitivity = 1;

    // Tail and aiming share the real angular-input helpers. Each axis applies
    // once per node, with any unhandled axis left for the gyro/output path.
    reset(); motion(100, -50);
    float h = ov_camera_axis_mouse((uintptr_t)aim, 0, 2.f, 30.f);
    float v = ov_camera_axis_mouse((uintptr_t)aim, 1, 1.f, 30.f);
    near(h, -4); near(v, -2);
    near(ov_camera_axis_mouse((uintptr_t)aim, 0, 2.f, 30.f), 2.f);
    angles[1] = v; angles[2] = h;
    F(aim, 0x18c) = F(aim, 0x194) = v; F(aim, 0x190) = h;
    aim_update(); near(angles[1], v); near(angles[2], h);
    reset(); motion(100, -50);
    angles[2] = F(aim, 0x190) = ov_camera_axis_mouse((uintptr_t)aim, 0, 0.f, 30.f);
    aim_update(); near(angles[1], -3); near(angles[2], -6);
    reset(); motion(100, -50);
    near(ov_camera_axis_mouse((uintptr_t)aim, 1, 0.f, 0.f), 0.f);
    aim_update(); near(angles[1], 0); near(angles[2], -6);
    reset(); shared[0x7db] = 0x10; shared[0x7d5] = 8; motion(100, -50);
    near(ov_camera_axis_mouse((uintptr_t)aim, 0, 2.f, 30.f), 2.f);
    aim_update(); near(angles[1], 0); near(angles[2], 0);
    reset(); *(uint64_t *)(shared + 0x218) = UINT64_C(0x40000000); motion(100, 0);
    near(ov_camera_axis_mouse((uintptr_t)aim, 0, 0.f, 30.f), -6.f);

    // Separate nodes of the same controller type must not steal one another's sample.
    reset(); motion(100, -50);
    assert(ov_camera_tail_mouse_idle((uintptr_t)aim, 1) == 0);
    assert(ov_camera_tail_mouse_idle((uintptr_t)aim, 1) == 0);
    near(ov_camera_axis_mouse((uintptr_t)aim, 0, 0.f, 1.f), -6.f);
    near(ov_camera_axis_mouse((uintptr_t)aim, 1, 0.f, 1.f), -3.f);
    motion(0, 0); assert(ov_camera_tail_mouse_idle((uintptr_t)aim, 1) == 1);
    assert(ov_camera_tail_mouse_idle((uintptr_t)aim, 0) == 0);
    shared[0x7db] = 0x10; shared[0x7d5] = 8; motion(100, 0);
    assert(ov_camera_tail_mouse_idle((uintptr_t)aim, 1) == 1);
    shared[0x7db] = shared[0x7d5] = 0; captured = 0; motion(100, 0);
    assert(ov_camera_tail_mouse_idle((uintptr_t)aim, 1) == 1); captured = 1;

    _Alignas(8) uint8_t other[0x250] = {0};
    *(uint8_t **)(other + 0x20) = shared;
    reset(); motion(100, 0);
    near(ov_camera_axis_mouse((uintptr_t)aim, 0, 0.f, 30.f), -6.f);
    near(ov_camera_axis_mouse((uintptr_t)other, 0, 0.f, 30.f), -6.f);

    // A fixed mouse distance turns by the same amount at different update rates.
    for (int n = 1; n <= 20; n *= 2) {
        reset();
        for (int i = 0; i < n; i++) { motion(160 / n, 0); aim_update(); }
        near(angles[2], -9.6f); near(F(aim, 0x190), -9.6f);
    }
    puts("camera mouse: transitions, additive angles, limits, capture and frame-rate invariance passed");
    return 0;
}
