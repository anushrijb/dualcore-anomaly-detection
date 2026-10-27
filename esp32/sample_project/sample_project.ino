/* ===========================================================================
 * Sample project (Review 1) - ESP32 dual-core motor protection
 *
 *   DC motor + L298N, MPU6050 (vibration), A3144 Hall (RPM), DHT22 (T, RH),
 *   ACS712 (current), INMP441 (sound), buzzer + LED, Wi-Fi to the fog gateway.
 *
 *   Core 0  sensorTask (1 kHz)  : read every sensor, stamp a sequence number,
 *                                 queue the tick for Core 1
 *           dhtTask (0.5 Hz)    : DHT22 (slow, blocking ~5 ms) - kept off the 1 kHz path
 *           netTask             : store-and-forward uplink to the fog (HTTP POST + ACK)
 *   Core 1  dspTask             : windows, 18 features, multi-input Z-score model
 *                                 (learns the healthy motor for LEARN_S), staged
 *                                 control: warning -> step speed down -> stop
 *
 * The C files next to this sketch (ms_core, features, detector, mpu6050_sim)
 * are the SAME files used and tested in the PC simulation (csrc/).
 *
 * Arduino-ESP32 core 3.x. Libraries: "DHT sensor library" (Adafruit) +
 * "Adafruit Unified Sensor".  Board: "ESP32 Dev Module".
 * ======================================================================== */
#include <Arduino.h>
#include <Wire.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <ESP_I2S.h>
#include <DHT.h>

extern "C" {
#include "ms_core.h"
#include "dsp_features.h"
#include "detector.h"
}

/* ------------------------------------------------------------- settings -- */
const char *WIFI_SSID = "your-ssid";
const char *WIFI_PASS = "your-password";
const char *FOG_URL   = "http://192.168.1.10:8060/api/ingest";   /* python ui/fog_server.py --ingest */
const char *DEVICE_ID = "MOTOR-01";

const float LEARN_S   = 300.0f;      /* on-device learning of the healthy motor (s)  */
const float CENTER    = 3.0f;        /* Z-score threshold centre (from calibration)  */
const float SENS      = 3.0f;
const float RPM_NOM   = 3000.0f;     /* your motor's speed at 100 % duty             */

/* ----------------------------------------------------------------- pins -- */
const int PIN_SDA = 21, PIN_SCL = 22;                 /* MPU6050                  */
const int PIN_HALL = 18;                              /* A3144, open collector    */
const int PIN_DHT = 4;                                /* DHT22                    */
const int PIN_ACS = 35;                               /* ACS712 via divider, ADC1 */
const int PIN_I2S_SCK = 14, PIN_I2S_WS = 15, PIN_I2S_SD = 32;   /* INMP441       */
const int PIN_ENA = 25, PIN_IN1 = 26, PIN_IN2 = 27;   /* L298N                    */
const int PIN_BUZZER = 13, PIN_LED = 2;

const uint8_t MPU_ADDR = 0x68;
const float ACS_MV_PER_A = 185.0f;                    /* ACS712-05B               */
const float ACS_DIVIDER = 1.5f;                       /* 5 V -> 3.3 V divider     */

/* ---------------------------------------------------------- shared state -- */
static QueueHandle_t tickQ;                           /* Core 0 -> Core 1         */
static volatile uint32_t hallEdgeUs[16];
static volatile uint8_t hallHead = 0, hallTail = 0;
static volatile float dhtT = NAN, dhtRH = NAN;
static volatile bool dhtNew = false;
static volatile float gDuty = 1.0f;
static volatile uint32_t lostQueue = 0, lostLate = 0;  /* zero-loss counters        */
static float acsZeroV = 2.5f;

DHT dht(PIN_DHT, DHT22);
I2SClass i2s;

/* store-and-forward ring of packets for the fog */
struct Packet { uint32_t win, seqLast, nNew; uint8_t sim, state; float duty, vib, tMotor, rh, rpm, cur; };
static const int SF_CAP = 512;                        /* ~65 s of windows        */
static Packet sfRing[SF_CAP];
static volatile int sfHead = 0, sfCount = 0;
static volatile uint32_t sfDropped = 0;
static portMUX_TYPE sfMux = portMUX_INITIALIZER_UNLOCKED;

/* ------------------------------------------------------------- helpers -- */
void IRAM_ATTR hallISR() {
  uint8_t n = (hallHead + 1) & 15;
  if (n != hallTail) { hallEdgeUs[hallHead] = micros(); hallHead = n; }
}

static void setDuty(float d) {
  gDuty = d;
  ledcWrite(PIN_ENA, (uint32_t)(d * 255.0f));
  digitalWrite(PIN_IN1, d > 0 ? HIGH : LOW);
  digitalWrite(PIN_IN2, LOW);
}

static bool readMPU(int16_t out[7]) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);                                   /* ACCEL_XOUT_H */
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((int)MPU_ADDR, 14) != 14) return false;
  for (int i = 0; i < 7; i++) out[i] = (int16_t)((Wire.read() << 8) | Wire.read());
  return true;
}

/* ------------------------------------------------- Core 0: 1 kHz sensors -- */
void sensorTask(void *) {
  TickType_t last = xTaskGetTickCount();
  uint32_t seq = 0, t0 = micros();
  int32_t raw32[MS_AUD];
  for (;;) {
    vTaskDelayUntil(&last, pdMS_TO_TICKS(1));        /* 1 kHz (configTICK_RATE_HZ = 1000) */
    uint32_t start = micros();
    uint32_t due = t0 + seq * 1000u;
    if ((int32_t)(start - due) > 85000) lostLate++;   /* later than the MPU6050 FIFO depth */

    ms_tick_t tk;
    memset(&tk, 0, sizeof tk);
    tk.seq = seq++;
    readMPU(tk.mpu);
    float v = analogReadMilliVolts(PIN_ACS) / 1000.0f * ACS_DIVIDER;
    tk.current_a = (v - acsZeroV) * 1000.0f / ACS_MV_PER_A;
    size_t got = i2s.readBytes((char *)raw32, sizeof raw32);   /* 8 samples at 8 kHz */
    for (int k = 0; k < MS_AUD; k++) tk.audio[k] = (k * 4 < (int)got) ? (int16_t)(raw32[k] >> 14) : 0;
    if (hallTail != hallHead) {                       /* Hall edge(s) since last tick */
      tk.hall = 1;
      int32_t off = (int32_t)(hallEdgeUs[hallTail] - start) + 1000;   /* position inside the last 1 ms */
      tk.hall_us = (float)(off < 0 ? 0 : (off > 999 ? 999 : off));
      hallTail = (hallTail + 1) & 15;
    }
    tk.dht_new = dhtNew; dhtNew = false;
    tk.dht_t = dhtT; tk.dht_rh = dhtRH;
    if (xQueueSend(tickQ, &tk, 0) != pdTRUE) lostQueue++;
  }
}

void dhtTask(void *) {
  for (;;) {
    float t = dht.readTemperature(), h = dht.readHumidity();
    if (!isnan(t) && !isnan(h)) { dhtT = t; dhtRH = h; dhtNew = true; }
    vTaskDelay(pdMS_TO_TICKS(2000));
  }
}

/* -------------------------------------------- Core 1: detect + control -- */
static ms_windower_t WD;
static ms_window_t W;
static detector_t DET;
static ms_ctrl_t CTRL;

void dspTask(void *) {
  ms_tick_t tk;
  uint32_t win = 0, rxAtLast = 0;
  for (;;) {
    if (xQueueReceive(tickQ, &tk, portMAX_DELAY) != pdTRUE) continue;
    if (!ms_windower_push(&WD, &tk, gDuty, &W)) continue;
    float f[MS_NF], raw[6];
    float t = (tk.seq + 1) / (float)MS_FS;
    ms_features(&W, RPM_NOM, f);
    ms_raw_values(&W, f, raw);
    int sim = -1;
    if (t <= LEARN_S) { det_learn(&DET, f); ms_ctrl_learn(&CTRL, raw); }
    else {
      if (!DET.ready) det_finalize(&DET);
      sim = det_similarity(&DET, f);
      setDuty(ms_ctrl_step(&CTRL, t, raw, sim));
    }
    digitalWrite(PIN_LED, CTRL.state != ST_NORMAL);
    digitalWrite(PIN_BUZZER, CTRL.state == ST_WARNING || CTRL.state == ST_STOPPED);

    Packet p = {win++, tk.seq, (uint32_t)(WD.received - rxAtLast), (uint8_t)(sim < 0 ? 255 : sim),
                (uint8_t)CTRL.state, gDuty, raw[0], raw[2], raw[4], raw[1] * RPM_NOM * W.duty, raw[5]};
    rxAtLast = WD.received;
    portENTER_CRITICAL(&sfMux);
    if (sfCount == SF_CAP) { sfHead = (sfHead + 1) % SF_CAP; sfCount--; sfDropped++; }
    sfRing[(sfHead + sfCount) % SF_CAP] = p;
    sfCount++;
    portEXIT_CRITICAL(&sfMux);
  }
}

/* --------------------------------------- Core 0: store-and-forward uplink -- */
void netTask(void *) {
  HTTPClient http;
  char body[320];
  for (;;) {
    if (sfCount == 0 || WiFi.status() != WL_CONNECTED) { vTaskDelay(pdMS_TO_TICKS(20)); continue; }
    Packet p = sfRing[sfHead];
    snprintf(body, sizeof body,
             "{\"device\":\"%s\",\"win\":%u,\"seq_last\":%u,\"n_new\":%u,\"sim\":%d,\"state\":%u,\"duty\":%.2f,"
             "\"vib\":%.4f,\"t_motor\":%.2f,\"rh\":%.1f,\"rpm\":%.0f,\"current\":%.3f,"
             "\"gaps\":%llu,\"lost_queue\":%u,\"lost_late\":%u,\"sf_dropped\":%u}",
             DEVICE_ID, p.win, p.seqLast, p.nNew, p.sim == 255 ? -1 : p.sim, p.state, p.duty, p.vib, p.tMotor,
             p.rh, p.rpm, p.cur, (unsigned long long)WD.gaps, lostQueue, lostLate, sfDropped);
    http.begin(FOG_URL);
    http.addHeader("Content-Type", "application/json");
    int code = http.POST((uint8_t *)body, strlen(body));
    http.end();
    if (code == 200) {                                /* ACK: remove from buffer */
      portENTER_CRITICAL(&sfMux);
      sfHead = (sfHead + 1) % SF_CAP; sfCount--;
      portEXIT_CRITICAL(&sfMux);
    } else {
      vTaskDelay(pdMS_TO_TICKS(200));                 /* no ACK: keep, retry later */
    }
  }
}

/* ---------------------------------------------------------------- setup -- */
void setup() {
  Serial.begin(115200);
  pinMode(PIN_IN1, OUTPUT); pinMode(PIN_IN2, OUTPUT);
  pinMode(PIN_BUZZER, OUTPUT); pinMode(PIN_LED, OUTPUT);
  ledcAttach(PIN_ENA, 20000, 8);                      /* 20 kHz PWM, 8-bit */
  setDuty(1.0f);

  Wire.begin(PIN_SDA, PIN_SCL, 400000);
  Wire.beginTransmission(MPU_ADDR); Wire.write(0x6B); Wire.write(0x00); Wire.endTransmission();  /* wake  */
  Wire.beginTransmission(MPU_ADDR); Wire.write(0x1A); Wire.write(0x01); Wire.endTransmission();  /* DLPF  */
  Wire.beginTransmission(MPU_ADDR); Wire.write(0x19); Wire.write(0x00); Wire.endTransmission();  /* 1 kHz */

  pinMode(PIN_HALL, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(PIN_HALL), hallISR, FALLING);
  dht.begin();
  i2s.setPins(PIN_I2S_SCK, PIN_I2S_WS, -1, PIN_I2S_SD);
  i2s.begin(I2S_MODE_STD, 8000, I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO);

  analogReadResolution(12);
  float z = 0;                                        /* ACS712 zero-current offset */
  for (int i = 0; i < 200; i++) { z += analogReadMilliVolts(PIN_ACS) / 1000.0f * ACS_DIVIDER; delay(1); }
  acsZeroV = z / 200;

  fft_init();
  det_init(&DET, DET_ZSM, MS_NF, CENTER, SENS);
  ms_detector_setup(&DET);
  ms_ctrl_init(&CTRL, CTRL_AI);
  memset(&WD, 0, sizeof WD);

  WiFi.begin(WIFI_SSID, WIFI_PASS);
  tickQ = xQueueCreate(512, sizeof(ms_tick_t));       /* 512 ticks = 0.5 s of slack */

  xTaskCreatePinnedToCore(sensorTask, "sensors", 4096, NULL, 5, NULL, 0);
  xTaskCreatePinnedToCore(dhtTask, "dht", 3072, NULL, 1, NULL, 0);
  xTaskCreatePinnedToCore(netTask, "net", 6144, NULL, 2, NULL, 0);
  xTaskCreatePinnedToCore(dspTask, "dsp", 16384, NULL, 4, NULL, 1);
  Serial.println("Learning the healthy motor for 300 s...");
}

void loop() {
  static uint32_t last = 0;
  if (millis() - last > 5000) {                       /* status + zero-loss counters */
    last = millis();
    Serial.printf("state=%d duty=%.2f gaps=%llu lost_queue=%u lost_late=%u sf=%d sf_dropped=%u\n",
                  CTRL.state, gDuty, (unsigned long long)WD.gaps, lostQueue, lostLate, sfCount, sfDropped);
  }
  delay(10);
}
