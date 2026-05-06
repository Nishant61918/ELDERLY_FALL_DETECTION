#pragma once

// ============================================================================
//  ELDERLY FALL DETECTION SYSTEM — CONFIGURATION
//  Team: Bishal, Nishant, Sabin | Kathford College
//
//  Every constant used by fall_detection_esp32.ino is defined here so you can
//  tune the system for a new environment without touching the main sketch.
//
//  ⚠  SECURITY NOTE
//     This file contains a WiFi password and an IFTTT API key in plaintext.
//     For any deployment beyond a classroom demo:
//       1. Create a separate secrets.h file with the credentials.
//       2. Add secrets.h to .gitignore so it is never committed.
//       3. Replace the literals below with:
//            #include "secrets.h"   // defines WIFI_SSID, WIFI_PASSWORD, IFTTT_KEY
// ============================================================================

// ----------------------------------------------------------------------------
//  WiFi credentials
// ----------------------------------------------------------------------------
//  The ESP32 connects to this network on startup to send IFTTT notifications.
//  If the network is unavailable the system falls back to local-only alerts
//  (buzzer + LED) and retries the connection every 30 s non-blocking.
// ----------------------------------------------------------------------------
#define WIFI_SSID       "Bishal"
#define WIFI_PASSWORD   "bishal123"

// ----------------------------------------------------------------------------
//  IFTTT Maker Webhooks
// ----------------------------------------------------------------------------
//  When a fall is confirmed the sketch sends an HTTP GET to:
//    http://maker.ifttt.com/trigger/<IFTTT_EVENT>/with/key/<IFTTT_KEY>
//  with three query parameters:
//    value1 = "Fall Alert"
//    value2 = CNN confidence (%)
//    value3 = peak acceleration magnitude (m/s²)
//
//  To set up: go to ifttt.com → Create → If Webhooks → Then your action
//  (e.g. send an email or SMS).  Copy the Webhooks key from
//  https://ifttt.com/maker_webhooks/settings into IFTTT_KEY below.
// ----------------------------------------------------------------------------
#define IFTTT_HOST      "maker.ifttt.com"
#define IFTTT_EVENT     "Fall_Detected"
#define IFTTT_KEY       "fBj4ufiein6Obt0SsLzXK0oKx66q9Z2x4CsqGBNR4Vo"

// ----------------------------------------------------------------------------
//  ESP32 GPIO assignments
// ----------------------------------------------------------------------------
//  Breadboard wiring:
//    MPU-6050 SDA → GPIO 21   (ESP32 default I²C SDA)
//    MPU-6050 SCL → GPIO 22   (ESP32 default I²C SCL)
//    MPU-6050 VCC → 3.3 V,  GND → GND
//    Buzzer (+)   → GPIO 4    (passive buzzer; use 100 Ω series resistor)
//    LED anode    → GPIO 2    (built-in LED on most DevKit V1 boards; 330 Ω series)
// ----------------------------------------------------------------------------
#define SDA_PIN         21
#define SCL_PIN         22
#define BUZZER_PIN      4
#define LED_PIN         2

// ----------------------------------------------------------------------------
//  MPU-6050 I²C address
// ----------------------------------------------------------------------------
//  AD0 pin LOW  → 0x68 (default, used here)
//  AD0 pin HIGH → 0x69 (use if two MPU-6050s share the same bus)
// ----------------------------------------------------------------------------
#define MPU_ADDR        0x68

// ----------------------------------------------------------------------------
//  MPU-6050 conversion constants
// ----------------------------------------------------------------------------
//  These match the full-scale ranges configured in setupMPU() and the ranges
//  used when the SisFall dataset was collected.  Changing either the register
//  value in setupMPU() OR these constants without updating the other will break
//  the normalisation.
//
//  ACCEL_SCALE : 2048 LSB per g  → full-scale ±16 g  (AFS_SEL = 3)
//  GYRO_SCALE  : 16.4 LSB per °/s → full-scale ±2000 °/s  (FS_SEL = 3)
//  G_TO_MS2    : 9.80665 m/s² per g  (standard gravity)
//  DEG_TO_RAD  : π / 180  (converts degrees to radians)
// ----------------------------------------------------------------------------
#define ACCEL_SCALE     2048.0f
#define GYRO_SCALE      16.4f
#define G_TO_MS2        9.80665f
#define DEG_TO_RAD      0.017453292f

// ----------------------------------------------------------------------------
//  Sampling and windowing
// ----------------------------------------------------------------------------
//  SAMPLE_RATE_HZ  : Target sensor sampling rate.  The MPU-6050 internal ODR
//                    is 1 kHz; at 200 Hz we read every 5th sample.
//
//  READ_INTERVAL   : milliseconds between sensor reads = 1000 / SAMPLE_RATE_HZ.
//                    The main loop triggers a read whenever millis() advances
//                    by at least this many ms (non-blocking, no delay()).
//
//  WINDOW_SIZE     : Number of samples fed to the CNN in one inference.
//                    MUST match the model's input shape (dim 1).
//                    With 400 samples at 200 Hz this is exactly 2 seconds of data.
//
//                    ⚠ Do NOT shrink WINDOW_SIZE without retraining the model.
//                    The model was trained on 400-sample SisFall windows; feeding
//                    a shorter window will cause garbage output.
//
//  NUM_CHANNELS    : 6 sensor channels per sample: ax, ay, az (m/s²), gx, gy, gz (rad/s).
//                    Must match the model's input shape (dim 2) and the number of
//                    entries in CHANNEL_MEAN[] / CHANNEL_STD[] below.
// ----------------------------------------------------------------------------
#define SAMPLE_RATE_HZ      200
#define READ_INTERVAL       5           // ms  (= 1000 / SAMPLE_RATE_HZ)
#define WINDOW_SIZE         400         // samples  (= 2.0 s at 200 Hz)
#define NUM_CHANNELS        6           // ax ay az gx gy gz

// ----------------------------------------------------------------------------
//  Calibration
// ----------------------------------------------------------------------------
//  CALIBRATION_SAMPLES : Number of readings averaged to compute DC offsets.
//                        500 samples at ~2 ms each ≈ 1 s of data.
//
//  CALIBRATION_DELAY   : ms between calibration readings.  Keeps the
//                        sampling rate slower than the MPU-6050 output rate
//                        so consecutive reads are independent.
// ----------------------------------------------------------------------------
#define CALIBRATION_SAMPLES 500
#define CALIBRATION_DELAY   2           // ms

// ----------------------------------------------------------------------------
//  TensorFlow Lite Micro
// ----------------------------------------------------------------------------
//  TENSOR_ARENA_SIZE : Byte size of the static memory block given to TFLite
//                      for storing model weights, activations, and temporaries.
//                      If AllocateTensors() fails, increase this value.
//                      Call interpreter->arena_used_bytes() at runtime to find
//                      the exact minimum; the printout in setupTFLite() does this.
//                      40 000 bytes is safe for the 54 536-byte SisFall 1D-CNN.
//
//  INPUT_SCALE /       : Quantisation parameters from the .tflite export.
//  INPUT_ZERO_POINT      They define how float activations are mapped to int8:
//                          float_value = (int8_value − zero_point) × scale
//                        INPUT_SCALE ≈ 1/25.5 gives ~6 quantisation steps per σ.
//
//  OUTPUT_SCALE /      : Same parameters for the output tensor.
//  OUTPUT_ZERO_POINT     OUTPUT_ZERO_POINT = −128 means the int8 range [−128, 127]
//                        maps to [0.0, 0.996] in float — i.e. unsigned probability.
// ----------------------------------------------------------------------------
#define TENSOR_ARENA_SIZE   40000

#define INPUT_SCALE         0.0392157f   // ≈ 1/25.5
#define INPUT_ZERO_POINT    0
#define OUTPUT_SCALE        0.00390625f  // = 1/256
#define OUTPUT_ZERO_POINT   -128

// ----------------------------------------------------------------------------
//  Z-score normalisation — SisFall per-channel statistics
// ----------------------------------------------------------------------------
//  These values were computed from the full SisFall training split and are the
//  same statistics used during model training.  They MUST NOT be changed unless
//  the model is retrained with different normalisation.
//
//  Physics context:
//    ax mean ≈ 0      (chest-worn: X axis mostly perpendicular to gravity)
//    ay mean ≈ −6.8   (Y axis carries most of the gravity component at ±16 g scale)
//    az mean ≈ −1.1   (Z axis, small gravity projection)
//    gyro means ≈ 0   (sensor is on average not rotating during the full dataset)
//
//  Standard deviations reflect the diversity of movements in SisFall:
//    accel STDs ≈ 4–6 m/s² — wide range covering falls and ADLs
//    gyro  STDs ≈ 0.4–0.6 rad/s
// ----------------------------------------------------------------------------
const float CHANNEL_MEAN[NUM_CHANNELS] = {
    -0.0461f,    // ax  (m/s²)
    -6.8318f,    // ay  (m/s²)
    -1.1165f,    // az  (m/s²)
    -0.01996f,   // gx  (rad/s)
    -0.74466f,   // gy  (rad/s)
    -0.06945f    // gz  (rad/s)
};

const float CHANNEL_STD[NUM_CHANNELS] = {
    4.0721f,     // ax
    5.8187f,     // ay
    4.8141f,     // az
    0.44048f,    // gx
    0.62880f,    // gy
    0.54259f     // gz
};

// ----------------------------------------------------------------------------
//  Fall detection thresholds and debounce
// ----------------------------------------------------------------------------
//
//  FALL_THRESHOLD
//    CNN fall probability must equal or exceed this value for a window to be
//    flagged as a "candidate".
//    → 0.60 gives 15 % headroom below 0.75 to compensate for the breadboard
//      drop signature differing slightly from a body-worn SisFall recording
//      (no body-mass damping, different axis orientation).
//    → Raise toward 0.75 if false positives appear during normal use.
//    → Lower toward 0.50 only if genuine falls are being missed.
//
//  MIN_ACCEL_MAGNITUDE
//    Peak ||a|| (m/s²) anywhere in the 2 s window must reach this value.
//    Acts as a physics sanity gate: slow rotations and arm swings cannot
//    produce a high CNN score without also generating a large acceleration.
//    → 11.0 m/s² is slightly above 1 g (9.81 m/s²) so the stationary sensor
//      does not false-trigger, while a drop from ~30 cm easily exceeds it.
//    → Lower to 10.0 if you are testing with very gentle simulated falls.
//    → Raise to 13.0+ for a production wearable on an active user.
//
//  CONSECUTIVE_DETECTIONS
//    Number of the last 3 overlapping windows that must both pass FALL_THRESHOLD
//    and MIN_ACCEL_MAGNITUDE before a fall alert is issued.
//    → 2 means two consecutive 2 s windows (offset by 1 s) must both flag a
//      candidate — effectively requiring the fall signature to span ~3 s total.
//    → Set to 1 if the system misses short/sharp falls that resolve in one window.
//    → Set to 3 (all windows) only if false positives are a serious problem.
//
//  ALERT_DURATION
//    Time in milliseconds the buzzer and LED stay active after a confirmed fall.
//    The system will not re-trigger during this period.
//    → 8000 ms (8 s) is enough for a demo; increase to 30 000+ for real deployment
//      so there is time for a caregiver to respond.
//
//  INFERENCE_TASK_STACK
//    FreeRTOS stack depth in bytes for the Core 1 inference task.
//    10 240 bytes is sufficient for TFLite Micro + Z-score normalisation + Serial.
//    Increase to 12 288 if you see stack overflow crashes (guru meditation error).
// ----------------------------------------------------------------------------
#define FALL_THRESHOLD          0.60f
#define MIN_ACCEL_MAGNITUDE     11.0f   // m/s²
#define CONSECUTIVE_DETECTIONS  2
#define ALERT_DURATION          8000    // ms
#define INFERENCE_TASK_STACK    10240   // bytes
