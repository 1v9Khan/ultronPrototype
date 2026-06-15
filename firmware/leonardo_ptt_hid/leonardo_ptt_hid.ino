/*
 * leonardo_ptt_hid.ino  --  HID-ONLY auto push-to-talk (hardened variant).
 *
 * Same job as leonardo_ptt, but with NO CDC serial port. The board enumerates
 * as a PURE HID composite: a Boot KEYBOARD interface (presses the team-PTT key)
 * + a vendor RAW HID collection (usage page 0xFFC0) for host commands -- exactly
 * the shape of a normal gaming keyboard with a config/RGB channel. Removing the
 * COM port eliminates the "HID keyboard that ALSO exposes a serial port"
 * composite signature that kernel anticheats associate with Arduino cheat
 * devices, and a custom USB VID/PID (set at COMPILE time, below) drops the
 * Arduino fingerprint. The host (kenning.ptt SerialHidPttBackend -> RawHid
 * backend) sends one-byte commands as HID OUTPUT reports via hidapi instead of
 * pyserial -- writing an HID output report is device I/O, NOT synthetic input.
 *
 * WIRE PROTOCOL (host -> device): a 64-byte raw-HID output report; byte[0] is:
 *     'D' key DOWN     'U' key UP     'H' heartbeat (refreshes the deadman)
 *
 * HARDWARE DEADMAN: auto-releases the key if no report arrives for DEADMAN_MS
 * while holding -- a host crash / unplug cannot jam the mic open.
 *
 * BUILD (CDC off + custom USB identity), from the repo root:
 *   arduino-cli compile --fqbn arduino:avr:leonardo \
 *     --build-property build.vid=0x1209 \
 *     --build-property build.pid=0xC0DE \
 *     --build-property 'build.usb_product="USB Keyboard"' \
 *     --build-property 'build.usb_manufacturer="Generic"' \
 *     --build-property 'build.extra_flags=-DCDC_DISABLED {build.usb_flags}' \
 *     firmware/leonardo_ptt_hid
 *
 * FLASHING NOTE: with CDC removed the board can no longer be serial-flashed.
 * The FIRST flash from the old CDC firmware still works (the 1200-baud touch
 * triggers the bootloader). For EVERY flash after that, double-press the RESET
 * button to enter the 8-second bootloader window, then upload.
 *
 * Requires the NicoHood "HID-Project" library (arduino-cli lib install HID-Project).
 * KEY must match the firmware keycode + your Valorant Team Voice bind. '6' here.
 */
#include <HID-Project.h>

const KeyboardKeycode KEY = KEY_6;       // this rig's Valorant team-PTT bind ('6')
const unsigned long DEADMAN_MS = 200;    // auto-release if host goes quiet while holding
const unsigned long MAX_HOLD_MS = 15000; // hard ceiling on a single continuous hold

uint8_t rawhidData[64];
bool holding = false;
unsigned long lastByteMs = 0;
unsigned long holdStartMs = 0;

void pressKey() {
  if (!holding) {
    BootKeyboard.press(KEY);
    holding = true;
    holdStartMs = millis();
    digitalWrite(LED_BUILTIN, HIGH);
  }
}

void releaseKey() {
  if (holding) {
    BootKeyboard.release(KEY);
    holding = false;
    digitalWrite(LED_BUILTIN, LOW);
  }
}

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);
  BootKeyboard.begin();
  RawHID.begin(rawhidData, sizeof(rawhidData));   // command channel (no CDC)
}

void loop() {
  // A host output report landed: byte 0 is the command.
  if (RawHID.available() > 0) {
    uint8_t cmd = rawhidData[0];
    lastByteMs = millis();
    if (cmd == 'D') {
      pressKey();
    } else if (cmd == 'H') {
      if (!holding) pressKey();   // recover a missed initial DOWN
    } else if (cmd == 'U') {
      releaseKey();
    }
    RawHID.enable();              // re-arm to accept the next report
  }

  // Hardware deadman + max-hold ceiling.
  if (holding) {
    unsigned long now = millis();
    if (now - lastByteMs > DEADMAN_MS) {
      releaseKey();
    } else if (MAX_HOLD_MS > 0 && now - holdStartMs > MAX_HOLD_MS) {
      releaseKey();
    }
  }
}
