/*
 * leonardo_ptt.ino  --  Auto push-to-talk for Ultron / Valorant team chat.
 *
 * Runs on an Arduino Leonardo (or any ATmega32u4 board: Pro Micro, etc.) that
 * enumerates as a NATIVE USB-HID keyboard. The PC host (kenning.ptt) sends one
 * byte over the USB serial port; this firmware presses/releases a REAL HID key.
 *
 * Why this exists / why it is anticheat-clean:
 *   Valorant TEAM voice chat is push-to-talk only (no voice activation). To let
 *   Ultron's relay callouts transmit, the team-PTT key must be held while he
 *   speaks. The PC host does NOTHING but write a byte to a COM port -- the
 *   actual keypress is produced HERE, by a physical USB keyboard, with NO
 *   "injected" flag. To Windows and to Vanguard this is indistinguishable from a
 *   real keyboard. No SendInput, no hooks, no injection on the PC side.
 *
 * Wire protocol (host -> board), one byte each:
 *   'D'  key DOWN      -- start holding KEY
 *   'U'  key UP        -- release KEY
 *   'H'  heartbeat     -- "still holding" (refreshes the deadman timer)
 *
 * HARDWARE DEADMAN (the critical failsafe):
 *   While the key is held, the host sends a heartbeat every ~50 ms. If this
 *   board stops receiving bytes for DEADMAN_MS (host crashed, USB unplugged,
 *   app killed), it AUTO-RELEASES the key. A stuck-open mic mid-match is the one
 *   real failure mode of auto-PTT; this kills it in hardware, independent of the
 *   PC. DEADMAN_MS must be comfortably larger than the host heartbeat interval
 *   (host 50 ms -> 200 ms here = 4x margin).
 *
 * SETUP:
 *   1. Set KEY below to the key you bind to "Team Voice" PTT in Valorant
 *      (Settings -> Audio -> Voice Chat). Default 'v'. Keep it in sync with
 *      config.yaml push_to_talk.key.
 *   2. Flash with the Arduino IDE (board: "Arduino Leonardo").
 *   3. Note the COM port it enumerates as and set push_to_talk.serial_port +
 *      push_to_talk.enabled in Ultron's config.
 *
 * The board NEVER presses anything on its own -- only in response to host bytes,
 * and it always releases on the deadman. It is comms-only (no gameplay input).
 */

#include <Keyboard.h>

// The in-game TEAM-voice push-to-talk key. Match your Valorant bind + config.
// '6' = this rig's Valorant Team Voice push-to-talk bind.
const char KEY = '6';

// Auto-release if no byte arrives for this long while holding (ms).
// Must be > host heartbeat interval (config push_to_talk.heartbeat_ms) with margin.
const unsigned long DEADMAN_MS = 200;

// Hard ceiling on a single continuous hold regardless of heartbeats (ms).
// A second failsafe above the deadman in case the host wedges while still
// somehow emitting heartbeats. 0 disables. (Host also has its own watchdog.)
const unsigned long MAX_HOLD_MS = 15000;

bool holding = false;
unsigned long lastByteMs = 0;
unsigned long holdStartMs = 0;

void pressKey() {
  if (!holding) {
    Keyboard.press(KEY);
    holding = true;
    holdStartMs = millis();
    digitalWrite(LED_BUILTIN, HIGH);
  }
}

void releaseKey() {
  if (holding) {
    Keyboard.release(KEY);
    holding = false;
    digitalWrite(LED_BUILTIN, LOW);
  }
}

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);
  Keyboard.begin();
  Serial.begin(9600);   // nominal; USB CDC runs full speed regardless
  // Boot safe: nothing held.
  holding = false;
}

void loop() {
  // Drain any received bytes.
  while (Serial.available() > 0) {
    int b = Serial.read();
    lastByteMs = millis();
    switch (b) {
      case 'D':  // down
        pressKey();
        break;
      case 'H':  // heartbeat -- keep holding; lastByteMs already refreshed
        if (!holding) {
          // A heartbeat with no prior DOWN still asserts intent-to-hold,
          // so we don't silently drop a clip if the first DOWN was missed.
          pressKey();
        }
        break;
      case 'U':  // up
        releaseKey();
        break;
      default:
        // Ignore anything else (line noise / stray bytes).
        break;
    }
  }

  // Hardware deadman: release if the host went quiet while we were holding.
  if (holding) {
    unsigned long now = millis();
    if (now - lastByteMs > DEADMAN_MS) {
      releaseKey();
    } else if (MAX_HOLD_MS > 0 && now - holdStartMs > MAX_HOLD_MS) {
      releaseKey();
    }
  }
}
