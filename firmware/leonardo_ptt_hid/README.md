# Hardened HID-only PTT firmware (Arduino Leonardo)

The hardened variant of `../leonardo_ptt`. Same job (hold the Valorant team-PTT
key while Ultron speaks), but the device presents as a **plain keyboard with a
vendor config channel** instead of an "Arduino with a serial port":

- **No CDC / no COM port** (`-DCDC_DISABLED`) — drops the HID-keyboard+serial
  composite that anticheat associates with Arduino cheat devices.
- **Custom USB identity** — VID `0x1209` (pid.codes open hardware), product
  `"USB Keyboard"`, manufacturer `"Generic"` — drops the Arduino `0x2341` name.
- **Boot keyboard + vendor Raw HID** (usage page `0xFFC0`, via the NicoHood
  HID-Project library) — the host sends `D`/`U`/`H` as HID **output reports**
  (device I/O, *not* synthetic input), exactly like a keyboard's config channel.

Net: at the USB level it's indistinguishable from a Corsair/Razer/QMK keyboard.
Picked by the [auto-PTT hardening research](../../) as the #1 *legitimate* (not
evasion) measure. Host side: `RawHidPttBackend` (`hidapi`), auto-selected by
`push_to_talk.backend: "auto"`.

## Building

Requires the `HID-Project` library (`arduino-cli lib install HID-Project`).
The `--clean` matters — without it arduino-cli reuses a cached core and the
**USB descriptor** (VID/CDC) won't pick up the flags even though the sketch does:

```
arduino-cli compile --clean --output-dir build --fqbn arduino:avr:leonardo \
  --build-property build.vid=0x1209 \
  --build-property build.pid=0xC0DE \
  --build-property 'build.usb_product="USB Keyboard"' \
  --build-property 'build.usb_manufacturer="Generic"' \
  --build-property 'build.extra_flags=-DCDC_DISABLED {build.usb_flags}' \
  firmware/leonardo_ptt_hid
```

## Flashing — once CDC is gone, there is no serial port to auto-reset

So the normal `arduino-cli upload` (1200-baud touch) **cannot** trigger the
bootloader. Flash with a manual **double-tap of the RESET button**:

1. Build the `.hex` (above) — `build/leonardo_ptt_hid.ino.hex`.
2. Run a poller that waits for the Caterina bootloader (VID `0x2341` PID `0x0036`)
   and runs `avrdude` the instant it appears — see the `ptt_test`/session
   tooling, or just: poll `serial.tools.list_ports` for `2341:0036`, then
   `avrdude -c avr109 -p atmega32u4 -P <bootport> -b 57600 -D -U flash:w:<hex>:i`.
3. **Double-tap RESET** on the Leonardo (two quick presses). The LED fades — that's
   the bootloader. **Add ~1.3 s of settle** after the port appears before avrdude,
   or it races the bootloader and fails (`programmer is not responding`). A
   fresh **replug** first improves reliability.

The very first flash *from the old CDC firmware* can still use the normal
auto-upload (that firmware has a COM port); only subsequent flashes need the
double-tap.

## Key

`KEY` in the sketch must match your Valorant Team Voice (Push to Talk) bind.
This rig: `KEY_6`.
