// Armadilha fotográfica — versão @@ID@@: Grove Vision AI V2 (YOLOv8n INT8 no Ethos-U55)
// comandado por um XIAO ESP32S3 Sense (microSD) montado no conector do módulo (I2C 0x62).
//
// Ciclo: XIAO em sono profundo → PIR acorda (ext0) → liga o Grove por um MOSFET →
// AT+INVOKE em rajada → porta de MIN_HITS invocações com a mesma espécie → JPEG
// (last_image, base64) e registros.csv no cartão do XIAO → desliga o Grove → dorme.
// O módulo consome ~0,35 W rodando ou parado (medição da Core Electronics), por isso
// ele só é alimentado durante o evento.
//
// Biblioteca: Seeed_Arduino_SSCMA (Gerenciador de Bibliotecas do Arduino IDE).
// Placa: "XIAO_ESP32S3" no pacote esp32 da Espressif; PSRAM: OPI PSRAM habilitada.
// NÃO COMPILADO neste trabalho. Conferir: pinos do XIAO, se x/y das caixas são centro ou
// canto (não documentado) e o tempo de partida do Grove após ligar a alimentação.

#include <Arduino.h>
#include <FS.h>
#include <SD.h>
#include <SPI.h>
#include <Seeed_Arduino_SSCMA.h>
#include <Wire.h>
#include <driver/rtc_io.h>
#include <esp_sleep.h>
#include <mbedtls/base64.h>

static const gpio_num_t PINO_PIR = GPIO_NUM_1;  // D0 (RTC GPIO, necessário para ext0)
static const int PINO_ENERGIA_GROVE = 2;        // D1 → porta do MOSFET da alimentação do Grove
static const bool GROVE_LIGA_EM_ALTO = true;    // depende do circuito (N ou P)
static const int SD_CS = 21;                    // cartão da placa de expansão Sense
static const int INVOCACOES = 8;
static const int MIN_HITS = 3;
static const uint8_t ESCORE_MIN = 35;           // 0–100 (AT+TSCORE); limiar do Modelo A
static const uint32_t PARTIDA_GROVE_MS = 5000;  // tempo máximo para o módulo responder
static const char *ROTULOS[] = {"anta", "tamandua_bandeira", "lobo_guara"};
static const int N_ROTULOS = 3;

RTC_DATA_ATTR uint32_t contador_eventos = 0;  // sobrevive ao sono profundo
SSCMA AI;

static void energia_grove(bool ligado) {
  digitalWrite(PINO_ENERGIA_GROVE, ligado == GROVE_LIGA_EM_ALTO ? HIGH : LOW);
}

static bool iniciar_grove() {
  energia_grove(true);
  const uint32_t inicio = millis();
  while (millis() - inicio < PARTIDA_GROVE_MS) {
    if (AI.begin(&Wire)) return true;
    delay(200);
  }
  return false;
}

static void dormir() {
  energia_grove(false);
  // PIR em nível alto acorda; o pull-down segura a linha durante o sono.
  esp_sleep_enable_ext0_wakeup(PINO_PIR, 1);
  esp_sleep_pd_config(ESP_PD_DOMAIN_RTC_PERIPH, ESP_PD_OPTION_ON);
  rtc_gpio_pullup_dis(PINO_PIR);
  rtc_gpio_pulldown_en(PINO_PIR);
  esp_deep_sleep_start();
}

static bool salvar_jpeg_base64(const String &b64, const String &caminho) {
  size_t tamanho = 0;
  const unsigned char *origem = reinterpret_cast<const unsigned char *>(b64.c_str());
  mbedtls_base64_decode(nullptr, 0, &tamanho, origem, b64.length());
  if (tamanho == 0) return false;
  uint8_t *jpeg = static_cast<uint8_t *>(ps_malloc(tamanho));
  if (jpeg == nullptr) return false;
  bool ok = mbedtls_base64_decode(jpeg, tamanho, &tamanho, origem, b64.length()) == 0;
  if (ok) {
    File arquivo = SD.open(caminho, FILE_WRITE);
    ok = arquivo && arquivo.write(jpeg, tamanho) == tamanho;
    arquivo.close();
  }
  free(jpeg);
  return ok;
}

static void processar_evento() {
  if (!SD.begin(SD_CS)) {
    Serial.println("Cartão SD ausente");
    return;
  }
  if (!iniciar_grove()) {
    Serial.println("Grove Vision AI V2 não respondeu");
    return;
  }
  contador_eventos++;
  const String evento = "e" + String(contador_eventos);
  SD.mkdir("/midias");
  const bool novo_log = !SD.exists("/registros.csv");
  File log = SD.open("/registros.csv", FILE_APPEND);
  if (novo_log) log.println("evento,timestamp_iso,arquivo,rotulo,probabilidade");

  int seguidos[N_ROTULOS] = {0};
  uint8_t melhor[N_ROTULOS] = {0};
  int confirmado = -1;
  bool houve_deteccao = false;
  String arquivos[INVOCACOES];
  int salvos = 0;
  for (int i = 0; i < INVOCACOES; i++) {
    if (AI.invoke(1, false, true) != CMD_OK) continue;
    bool presente[N_ROTULOS] = {false};
    for (auto &caixa : AI.boxes()) {
      if (caixa.target >= N_ROTULOS || caixa.score < ESCORE_MIN) continue;
      presente[caixa.target] = true;
      houve_deteccao = true;
      melhor[caixa.target] = max(melhor[caixa.target], caixa.score);
    }
    for (int c = 0; c < N_ROTULOS; c++) {
      seguidos[c] = presente[c] ? seguidos[c] + 1 : 0;
      if (confirmado < 0 && seguidos[c] >= MIN_HITS) confirmado = c;
    }
    const String caminho = "/midias/" + evento + "_" + String(i) + ".jpg";
    if (salvar_jpeg_base64(AI.last_image(), caminho)) arquivos[salvos++] = caminho.substring(1);
  }
  // Sem relógio de tempo real: o timestamp é o contador de eventos. Use um RTC (DS3231)
  // ou ajuste por NTP na instalação para datas reais (exigência do Camtrap DP).
  String rotulo = "vazio";
  float prob = 0.0f;
  if (confirmado >= 0) {
    rotulo = ROTULOS[confirmado];
    prob = melhor[confirmado] / 100.0f;
  } else if (houve_deteccao) {
    rotulo = "inconclusivo";
  }
  for (int i = 0; i < salvos; i++) {
    log.printf("%s,1970-01-01T00:00:00+00:00,%s,%s,%.4f\n", evento.c_str(), arquivos[i].c_str(),
               rotulo.c_str(), prob);
  }
  log.close();
  Serial.printf("%s: %s (%.2f), %d imagens\n", evento.c_str(), rotulo.c_str(), prob, salvos);
}

void setup() {
  Serial.begin(115200);
  pinMode(PINO_ENERGIA_GROVE, OUTPUT);
  energia_grove(false);
  if (esp_sleep_get_wakeup_cause() == ESP_SLEEP_WAKEUP_EXT0) {
    rtc_gpio_deinit(PINO_PIR);
    processar_evento();
  }
  dormir();
}

void loop() {}
