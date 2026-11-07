// Harness for the locally generated angular helpers. No game-derived bodies here.
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cpu.h"
#include "simd.h"
#include "nvn_api.h"
#include "platform/platform.h"

void f_7000623ac0(Cpu *c);
void f_7000623e70(Cpu *c);
GfxStats gfx_counters;
volatile int g_fnprof;
static int32_t raw_x, raw_y;
static int takes;
float input_mouse_sensitivity(void) { return 1.f; }
int ikb_captured(void) { return 1; } // Synthetic backend; the test process uses TOTK_INPUT=none.
void ikb_mouse_take(int32_t *x, int32_t *y) { *x=raw_x; *y=raw_y; raw_x=raw_y=0; takes++; }
void plat_mutex_lock(PlatMutex *m) { (void)m; }
void plat_mutex_unlock(PlatMutex *m) { (void)m; }
void fnprof_hit(uint64_t a, uint64_t lr, uint64_t x0) { (void)a; (void)lr; (void)x0; }
void rt_udf(Cpu *c, uint64_t pc) { (void)c; fprintf(stderr,"unexpected UDF: %llx\n",(unsigned long long)pc); abort(); }
// Isolate the real helper's ABI/return path from its game-specific sensitivity getters.
void f_7000623b60(Cpu *c) { FP_SETS(0, 1.f); }
void f_7000623f10(Cpu *c) { FP_SETS(0, 1.f); }
static void near(float a,float b) { assert(fabsf(a-b)<.001f); }
_Alignas(16) static uint8_t node[0x250], shared[0x800], a[0x400], b[0x400], d[0x400], e[0x400], stack[0x200];

static float invoke(void (*fn)(Cpu *), float stick, int vertical) {
    Cpu state={0}, *c=&state;
    c->x[0]=(uintptr_t)node;
    c->x[29]=0x12340000; c->x[30]=0x12345678;
    c->sp=(uintptr_t)(stack+sizeof stack);
    uint64_t old_sp=c->sp;
    c->v[8]=0x11111111; c->v[9]=0x22222222; c->v[10]=0x33333333;
    FP_SETS(0, 30.f); FP_SETS(1, 1.f/60.f);
    *(float *)(shared+0x538+(vertical?4:0))=stick;
    fn(c);
    assert(c->sp==old_sp && c->x[29]==0x12340000 && c->x[30]==0x12345678);
    assert((uint64_t)c->v[8]==0x11111111 && (uint64_t)c->v[9]==0x22222222 && (uint64_t)c->v[10]==0x33333333);
    return VSF32(0);
}

int main(void) {
    *(uint8_t **)(node+8)=a; *(uint8_t **)(a+8)=b;
    *(uint8_t **)(b+0x1f0)=d; *(uint8_t **)(d+0x270)=e;
    *(uint8_t **)(e+0x360)=e;
    *(uint8_t **)(node+0x20)=shared;
    raw_x=100; raw_y=-50; gfx_counters.frames=1;
    // Native stick is -0.125 degrees, mouse contributes -6 / -3 degrees.
    near(invoke(f_7000623e70,.5f,0),-6.125f);
    near(invoke(f_7000623ac0,.5f,1),-3.125f);
    assert(takes==1);
    // A second query in the same present retains native input without duplicating mouse movement.
    near(invoke(f_7000623e70,.5f,0),-.125f);
    near(invoke(f_7000623ac0,.5f,1),-.125f);
    gfx_counters.frames++;
    near(invoke(f_7000623e70,.5f,0),-.125f);
    puts("generated camera helpers: real return path, axis direction, stick sum and guest ABI passed");
    return 0;
}
