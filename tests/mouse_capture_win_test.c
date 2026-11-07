// Win32 cursor integration test, no game files or game input. Run with TOTK_INPUT=none.
#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include "input.h"
#include "input_win.h"
#include "input_wsys.h"

static HWND window;
static int allowed = 1;
HWND gfx_window_handle(void) { return window; }
int input_cursor_allowed(void) { return allowed; } // Test backend, independent of real keyboard/game input.
void rt_log(const char *fmt, ...) { (void)fmt; }
static LRESULT CALLBACK proc(HWND h, UINT m, WPARAM w, LPARAM l) {
    LRESULT result;
    if (input_window_msg(h, m, w, l, &result)) return result;
    return DefWindowProcW(h, m, w, l);
}

int main(void) {
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    int failed = 0;
    POINT saved;
    GetCursorPos(&saved);
    HWND previous = GetForegroundWindow();
    WNDCLASSW wc = {.lpfnWndProc = proc, .hInstance = GetModuleHandleW(NULL), .lpszClassName = L"MouseCaptureTest"};
    RegisterClassW(&wc);
    window = CreateWindowW(wc.lpszClassName, L"Mouse capture test", WS_OVERLAPPEDWINDOW,
                           100, 100, 640, 480, NULL, NULL, wc.hInstance, NULL);
#define CHECK(test, label) do { if (!(test)) { fprintf(stderr, "mouse capture: FAILED %s\n", label); failed = 1; goto done; } } while (0)
    CHECK(window, "create window");
    ShowWindow(window, SW_SHOW);
    SetForegroundWindow(window);
    CHECK(GetForegroundWindow() == window, "foreground required for integration test");
    SendMessageW(window, WM_ACTIVATEAPP, TRUE, 0);
    CHECK(ikb_captured(), "capture enabled");
    RECT client, clip;
    POINT origin = {0, 0}, pos;
    GetClientRect(window, &client);
    ClientToScreen(window, &origin);
    GetClipCursor(&clip);
    CHECK(clip.left == origin.x + client.right / 2 && clip.top == origin.y + client.bottom / 2 &&
          clip.right == clip.left + 1 && clip.bottom == clip.top + 1, "one interior pixel");
    // Even attempted moves to desktop corners must stay at the centre, away
    // from taskbar reveal edges and overlays elsewhere on the game window.
    for (int i = 0; i < 4; i++) {
        SetCursorPos(i & 1 ? GetSystemMetrics(SM_CXSCREEN) - 1 : 0,
                     i & 2 ? GetSystemMetrics(SM_CYSCREEN) - 1 : 0);
        GetCursorPos(&pos);
        if (pos.x != clip.left || pos.y != clip.top)
            fprintf(stderr, "edge %d: cursor %ld,%ld clip %ld,%ld-%ld,%ld\n", i, pos.x, pos.y,
                    clip.left, clip.top, clip.right, clip.bottom);
        CHECK(pos.x == clip.left && pos.y == clip.top, "desktop edge blocked");
    }
    SendMessageW(window, WM_KEYDOWN, VK_ESCAPE, 0);
    CHECK(!ikb_captured(), "escape releases");
    SendMessageW(window, WM_LBUTTONDOWN, 0, 0);
    CHECK(ikb_captured(), "client click recaptures");
    allowed = 0; ikb_cursor_update();
    CHECK(!ikb_captured(), "menu/disabled input releases");
    allowed = 1; ikb_cursor_update();
    CHECK(ikb_captured(), "menu close recaptures");
    input_cursor_suspend(1);
    CHECK(!ikb_captured(), "resize/fullscreen transition releases");
    input_cursor_suspend(0);
    CHECK(ikb_captured(), "transition end recaptures");
    SendMessageW(window, WM_ACTIVATEAPP, FALSE, 0);
    CHECK(!ikb_captured(), "focus loss releases");
    GetClipCursor(&clip);
    CHECK(clip.right - clip.left > 1 && clip.bottom - clip.top > 1, "desktop clip restored");
    SendMessageW(window, WM_ACTIVATEAPP, TRUE, 0);
    SetWindowPos(window, NULL, 140, 140, 800, 600, SWP_NOZORDER | SWP_NOACTIVATE);
    GetClientRect(window, &client); origin = (POINT){0, 0}; ClientToScreen(window, &origin);
    GetClipCursor(&clip);
    CHECK(clip.left == origin.x + client.right / 2 && clip.top == origin.y + client.bottom / 2,
          "pin follows move and resize");
done:
    if (window) DestroyWindow(window);
    ClipCursor(NULL);
    SetCursorPos(saved.x, saved.y);
    if (previous) SetForegroundWindow(previous);
    if (!failed) puts("mouse capture: pin, edges, Esc, menu, suspend, focus and resize passed");
    return failed;
}
