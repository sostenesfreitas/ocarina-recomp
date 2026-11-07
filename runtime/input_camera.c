#include <math.h>
#include <stdint.h>

#include "input.h"
#include "input_wsys.h"
#include "nvn_api.h"
#include "platform/platform.h"

// Ocarina: the mouse turns the camera itself. ov_camera_mouse is called every frame by the camera update (patched in by
// tools/oot/patch_generated.py) with the camera node, and adds the raw mouse motion to the camera's own angles, in
// degrees: yaw at node+0x1a4 and pitch at node+0x18c, the two angles the game integrates from the right stick, so the
// stick and the mouse add up and the game's own speed limit on the stick does not apply to the mouse. Once the hook has
// run, the mouse no longer goes through the simulated right stick.
#define CAM_YAW_OFF 0x1a4
#define CAM_PITCH_OFF 0x18c
#define CAM_PITCH_MIN (-85.f)   // the limits the game keeps the pitch in (looking up / down)
#define CAM_PITCH_MAX 89.f
#define CAM_DEG_PER_COUNT 0.06f
static int cam_mouse_hooked;

int input_camera_mouse_active(void) {
    return __atomic_load_n(&cam_mouse_hooked, __ATOMIC_RELAXED);
}

// Controllers can update together during a transition. Sample once per present
// and apply each axis once per node, including nodes using the shared helpers.
static PlatMutex camera_mouse_lock = PLAT_MUTEX_INIT;
static uint64_t camera_mouse_frame = UINT64_MAX;
static int32_t camera_mouse_dx, camera_mouse_dy;
static struct { uint64_t node; unsigned axes; } camera_mouse_used[32];
static unsigned camera_mouse_nodes;

static int camera_mouse_sample(uint64_t node, unsigned axes, int consume, float *dx, float *dy) {
    __atomic_store_n(&cam_mouse_hooked, 1, __ATOMIC_RELAXED);
    plat_mutex_lock(&camera_mouse_lock);
    if (!ikb_captured()) {
        int32_t discard_x, discard_y;
        ikb_mouse_take(&discard_x, &discard_y);
        camera_mouse_frame = UINT64_MAX;
        camera_mouse_nodes = 0;
        plat_mutex_unlock(&camera_mouse_lock);
        return 0;
    }
    uint64_t frame = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    if (frame != camera_mouse_frame) {
        ikb_mouse_take(&camera_mouse_dx, &camera_mouse_dy);
        camera_mouse_frame = frame;
        camera_mouse_nodes = 0;
    }
    unsigned i;
    for (i = 0; i < camera_mouse_nodes; i++) if (camera_mouse_used[i].node == node) break;
    if (i == camera_mouse_nodes) {
        if (i == sizeof camera_mouse_used / sizeof *camera_mouse_used) {
            plat_mutex_unlock(&camera_mouse_lock);
            return 0;
        }
        camera_mouse_used[i].node = node;
        camera_mouse_used[i].axes = 0;
        camera_mouse_nodes++;
    }
    unsigned pending = axes & ~camera_mouse_used[i].axes;
    float scale = input_mouse_sensitivity() * CAM_DEG_PER_COUNT;
    if (consume) camera_mouse_used[i].axes |= axes;
    *dx = pending & 1 ? (float)camera_mouse_dx * scale : 0.f;
    *dy = pending & 2 ? (float)camera_mouse_dy * scale : 0.f;
    int apply = scale > 0 && (*dx || *dy);
    plat_mutex_unlock(&camera_mouse_lock);
    return apply;
}

static int camera_mouse_delta(uint64_t node, unsigned axes, float *dx, float *dy) {
    return camera_mouse_sample(node, axes, 1, dx, dy);
}

// Tail tests shared+0x530 bit 4 (right-stick inactivity) to enter manual
// rotation. Raw mouse motion must count as activity too, before the game's
// existing transition initializes the manual angles from the current view.
// Peeking leaves both axes for the angular helpers later in that update.
int ov_camera_tail_mouse_idle(uint64_t node, int idle) {
    if (!node || !idle) return idle;
    uint8_t *shared = *(uint8_t **)(uintptr_t)(node + 0x20);
    if (!shared || ((shared[0x7db] & 0x10) && (shared[0x7d5] & 0x48))) return idle;
    float dx, dy;
    return camera_mouse_sample(node, 3, 0, &dx, &dy) ? 0 : idle;
}

static float camera_mouse_wrap(float angle) {
    angle = fmodf(angle, 360.f);
    if (angle > 180.f) angle -= 360.f;
    if (angle < -180.f) angle += 360.f;
    return angle;
}

static float camera_mouse_clamp(float angle, float lo, float hi) {
    return angle < lo ? lo : angle > hi ? hi : angle;
}

void ov_camera_mouse(uint64_t node) {
    float dx, dy;
    if (!node || !camera_mouse_delta(node, 3, &dx, &dy)) return;
    float *yaw = (float *)(uintptr_t)(node + CAM_YAW_OFF), *pitch = (float *)(uintptr_t)(node + CAM_PITCH_OFF);
    *yaw = camera_mouse_wrap(*yaw - dx);   // right turns the view right: the scene moves left
    *pitch = camera_mouse_clamp(*pitch + dy, CAM_PITCH_MIN, CAM_PITCH_MAX);
}

// The game's shared angular-input helpers are also used by Tail (climbing)
// and other camera modes. Add degrees after the stick curve and before each
// controller's limits/integration. An axis handled here must not be added a
// second time by the item-aiming output hook later in the same update.
float ov_camera_axis_mouse(uint64_t node, int axis, float stick_delta, float speed) {
    float dx, dy;
    if (!node || axis < 0 || axis > 1 || !isfinite(stick_delta)) return stick_delta;
    if (!camera_mouse_delta(node, 1u << axis, &dx, &dy)) return stick_delta;
    uint8_t *shared = *(uint8_t **)(uintptr_t)(node + 0x20);
    // Match the helpers' own muted-input branch, rather than imposing the
    // Aiming controller's mode flags on Tail and the other controllers.
    if (!shared || ((shared[0x7db] & 0x10) && (shared[0x7d5] & 0x48)) ||
        !isfinite(speed) || speed == 0.f) return stick_delta;
    return stick_delta + (axis ? dy : -dx);
}

// ExecutePlayerCameraAiming has its own angular state. Called before its
// distance/pitch/yaw feeds the camera transform, not after the rendered view.
// Shift the accumulated and smoothed state together so the next update keeps
// the mouse turn; the game's stick and gyro contributions remain in the angles.
void ov_camera_aim_mouse(uint64_t node, uint64_t angles_addr) {
    float dx, dy;
    if (!node || !angles_addr || !camera_mouse_delta(node, 3, &dx, &dy)) return;
    uint8_t *n = (uint8_t *)(uintptr_t)node;
    uint8_t *shared = *(uint8_t **)(n + 0x20);
    if (!shared || (*(uint64_t *)(shared + 0x218) & UINT64_C(0x20000040000000)) ||
        ((shared[0x7db] & 0x10) && (shared[0x7d5] & 0x48))) return;
    float *angles = (float *)(uintptr_t)angles_addr;
    float lo = *(float *)(n + 0x224), hi = *(float *)(n + 0x228);
    if (!isfinite(lo) || !isfinite(hi) || lo > hi || !isfinite(angles[1]) || !isfinite(angles[2])) return;
    float base_pitch = *(float *)(n + 0x194);
    float target_pitch = *(float *)(n + 0x18c);
    float base_yaw = *(float *)(n + 0x190);
    if (!isfinite(base_pitch) || !isfinite(target_pitch) || !isfinite(base_yaw)) return;
    int restricted = *(uint32_t *)(n + 0x23c) == 2;
    if (restricted) {
        lo = fmaxf(lo, -25.f);
        hi = fminf(hi, 5.f);
        if (lo > hi) return;
    }
    // The output may already include gyro beyond the base pitch limits. Limit
    // the mouse's contribution on the base, preserving that existing offset.
    float pitch_delta = dy ? camera_mouse_clamp(base_pitch + dy, lo, hi) - base_pitch : 0.f;
    float pitch = camera_mouse_clamp(angles[1] + pitch_delta, -89.9f, 89.9f);
    float yaw = camera_mouse_wrap(angles[2] - dx);
    if (restricted) yaw = camera_mouse_clamp(yaw, -12.f, 12.f);
    pitch_delta = pitch - angles[1];
    float yaw_delta = camera_mouse_wrap(yaw - angles[2]);
    *(float *)(n + 0x18c) = camera_mouse_clamp(target_pitch + pitch_delta, lo, hi);
    *(float *)(n + 0x194) = camera_mouse_clamp(base_pitch + pitch_delta, lo, hi);
    float state_yaw = camera_mouse_wrap(base_yaw + yaw_delta);
    *(float *)(n + 0x190) = restricted ? camera_mouse_clamp(state_yaw, -12.f, 12.f) : state_yaw;
    angles[1] = pitch;
    angles[2] = yaw;
}
