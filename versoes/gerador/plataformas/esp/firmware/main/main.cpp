// Armadilha fotográfica — versão @@ID@@: ESP32-S3-EYE como subsistema de captura
// (arquitetura C da pesquisa) com pré-classificação TinyML opcional.
//
// Ciclo: sono profundo (~7 µA no chip) → PIR acorda por ext0 → FOTOS JPEG UXGA no
// microSD → MobileNetV2 INT8 (ESP-DL) classifica o quadro inteiro de cada foto →
// linha em /sdcard/registros.csv → sono profundo. A pré-classificação NÃO decide
// descarte: tudo é gravado e a classificação definitiva vem do lote (versões 01c/08c/13c)
// ou da importação Camtrap DP (`python -m animal_monitor_hw import-csv`).
//
// NÃO COMPILADO neste trabalho (sem ESP-IDF no ambiente). APIs conferidas na
// documentação do ESP-DL 3.3, esp32-camera e ESP-IDF v5/v6; marcadas como CONFERIR onde a
// documentação não fecha o detalhe.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

#include "dl_image_preprocessor.hpp"
#include "dl_model_base.hpp"
#include "driver/rtc_io.h"
#include "driver/sdmmc_host.h"
#include "esp_camera.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_sleep.h"
#include "esp_vfs_fat.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "img_converters.h"
#include "sdmmc_cmd.h"

static const char *TAG = "armadilha";
// O guia do S3-EYE diz que todos os GPIOs estão em uso exceto o GPIO3 (LED verde,
// pino de strapping). O PIR deve ficar em nível baixo durante o reset (é o repouso do PIR).
static const gpio_num_t PINO_PIR = GPIO_NUM_3;
static const int FOTOS = 3;
static const int INTERVALO_MS = 300;
static const char *ROTULOS[] = {"anta", "tamandua_bandeira", "lobo_guara", "desconhecido"};
static const int N_ROTULOS = 4;
static const char *FUSO = "<-03>3";  // America/Sao_Paulo sem horário de verão

RTC_DATA_ATTR static uint32_t contador_eventos = 0;
extern const uint8_t modelo_espdl[] asm("_binary_classificador_espdl_start");

static camera_config_t configuracao_camera() {
  // Pinos do ESP32-S3-EYE v2.2 (camera_pins.h do arduino-esp32, iguais ao esp-bsp).
  camera_config_t c = {};
  c.pin_pwdn = -1;
  c.pin_reset = -1;
  c.pin_xclk = 15;
  c.pin_sccb_sda = 4;
  c.pin_sccb_scl = 5;
  c.pin_d7 = 16;
  c.pin_d6 = 17;
  c.pin_d5 = 18;
  c.pin_d4 = 12;
  c.pin_d3 = 10;
  c.pin_d2 = 8;
  c.pin_d1 = 9;
  c.pin_d0 = 11;
  c.pin_vsync = 6;
  c.pin_href = 7;
  c.pin_pclk = 13;
  c.xclk_freq_hz = 20000000;
  c.ledc_timer = LEDC_TIMER_0;
  c.ledc_channel = LEDC_CHANNEL_0;
  c.pixel_format = PIXFORMAT_JPEG;
  c.frame_size = FRAMESIZE_UXGA;  // 1600 × 1200, resolução nativa da OV2640
  c.jpeg_quality = 12;
  c.fb_count = 1;
  c.fb_location = CAMERA_FB_IN_PSRAM;
  c.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  return c;
}

static bool montar_sd(sdmmc_card_t **cartao) {
  // microSD do S3-EYE em SDMMC de 1 bit: CMD 38, CLK 39, D0 40 (esp-bsp).
  sdmmc_host_t host = SDMMC_HOST_DEFAULT();
  sdmmc_slot_config_t slot = SDMMC_SLOT_CONFIG_DEFAULT();
  slot.width = 1;
  slot.clk = GPIO_NUM_39;
  slot.cmd = GPIO_NUM_38;
  slot.d0 = GPIO_NUM_40;
  slot.flags |= SDMMC_SLOT_FLAG_INTERNAL_PULLUP;
  esp_vfs_fat_sdmmc_mount_config_t montagem = {};
  montagem.format_if_mount_failed = false;
  montagem.max_files = 5;
  montagem.allocation_unit_size = 16 * 1024;
  return esp_vfs_fat_sdmmc_mount("/sdcard", &host, &slot, &montagem, cartao) == ESP_OK;
}

static void ajustar_relogio_se_preciso() {
  // Sem RTC com bateria: o horário sobrevive ao sono profundo, não à falta de energia.
  // Na instalação, grave a data em /sdcard/relogio.txt (ex.: 2026-11-02T08:30:00).
  setenv("TZ", FUSO, 1);
  tzset();
  if (time(nullptr) > 1700000000) return;
  FILE *arquivo = fopen("/sdcard/relogio.txt", "r");
  if (arquivo == nullptr) return;
  struct tm data = {};
  if (fscanf(arquivo, "%d-%d-%dT%d:%d:%d", &data.tm_year, &data.tm_mon, &data.tm_mday,
             &data.tm_hour, &data.tm_min, &data.tm_sec) == 6) {
    data.tm_year -= 1900;
    data.tm_mon -= 1;
    struct timeval agora = {mktime(&data), 0};
    settimeofday(&agora, nullptr);
  }
  fclose(arquivo);
}

static void agora_iso(char *saida, size_t tamanho) {
  time_t agora = time(nullptr);
  struct tm local;
  localtime_r(&agora, &local);
  strftime(saida, tamanho, "%Y-%m-%dT%H:%M:%S-03:00", &local);
}

static int classificar(dl::Model *modelo, dl::image::ImagePreprocessor *pre,
                       const camera_fb_t *foto, float *probabilidade) {
  // JPEG UXGA → RGB565 em 1/8 (200 × 150) pelo decodificador do esp32-camera: barato e
  // suficiente para uma entrada de 128 × 128.
  const int largura = foto->width / 8;
  const int altura = foto->height / 8;
  uint8_t *rgb = static_cast<uint8_t *>(
      heap_caps_malloc(largura * altura * 2, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  if (rgb == nullptr) return -1;
  if (!jpg2rgb565(foto->buf, foto->len, rgb, JPG_SCALE_8X)) {
    heap_caps_free(rgb);
    return -1;
  }
  // CONFERIR: ordem de bytes do RGB565 do jpg2rgb565 (BE/LE). Cor trocada não quebra a
  // execução, mas piora a classificação; teste com model->test() e uma foto conhecida.
  dl::image::img_t imagem = {rgb, static_cast<uint16_t>(largura),
                             static_cast<uint16_t>(altura),
                             dl::image::DL_IMAGE_PIX_TYPE_RGB565BE};
  pre->preprocess(imagem);
  modelo->run();
  heap_caps_free(rgb);

  // Logits INT8 com expoente de potência de 2 (quantização do ESP-DL) → softmax.
  dl::TensorBase *saida = modelo->get_output();
  const int8_t *valores = saida->get_element_ptr<int8_t>();
  const float escala = std::ldexp(1.0f, saida->exponent);
  float logits[N_ROTULOS];
  float maximo = -1e9f;
  for (int i = 0; i < N_ROTULOS; i++) {
    logits[i] = valores[i] * escala;
    maximo = logits[i] > maximo ? logits[i] : maximo;
  }
  float soma = 0.0f;
  for (int i = 0; i < N_ROTULOS; i++) soma += (logits[i] = std::exp(logits[i] - maximo));
  int melhor = 0;
  for (int i = 1; i < N_ROTULOS; i++) melhor = logits[i] > logits[melhor] ? i : melhor;
  *probabilidade = logits[melhor] / soma;
  return melhor;
}

static void registrar_evento() {
  sdmmc_card_t *cartao = nullptr;
  if (!montar_sd(&cartao)) {
    ESP_LOGE(TAG, "microSD ausente");
    return;
  }
  ajustar_relogio_se_preciso();
  camera_config_t configuracao = configuracao_camera();
  if (esp_camera_init(&configuracao) != ESP_OK) {
    ESP_LOGE(TAG, "falha na câmera");
    return;
  }
  auto *modelo = new dl::Model(reinterpret_cast<const char *>(modelo_espdl),
                               fbs::MODEL_LOCATION_IN_FLASH_RODATA);
  modelo->minimize();
  // Normalização ImageNet igual à do treino (animal_monitor_b.augment.to_tensor).
  auto *pre = new dl::image::ImagePreprocessor(modelo, {123.675f, 116.28f, 103.53f},
                                               {58.395f, 57.12f, 57.375f});
  pre->enable_letterbox({114, 114, 114});  // CONFERIR assinatura; o treino usou 114

  contador_eventos++;
  mkdir("/sdcard/midias", 0775);
  const bool novo = access("/sdcard/registros.csv", F_OK) != 0;
  FILE *log = fopen("/sdcard/registros.csv", "a");
  if (novo && log) fprintf(log, "evento,timestamp_iso,arquivo,rotulo,probabilidade\n");
  for (int i = 0; i < FOTOS; i++) {
    camera_fb_t *foto = esp_camera_fb_get();
    if (foto == nullptr) continue;
    char arquivo[64];
    snprintf(arquivo, sizeof(arquivo), "midias/e%lu_%d.jpg",
             static_cast<unsigned long>(contador_eventos), i);
    char caminho[80];
    snprintf(caminho, sizeof(caminho), "/sdcard/%s", arquivo);
    FILE *jpeg = fopen(caminho, "wb");
    if (jpeg) {
      fwrite(foto->buf, 1, foto->len, jpeg);
      fclose(jpeg);
    }
    float prob = 0.0f;
    const int indice = classificar(modelo, pre, foto, &prob);
    char quando[32];
    agora_iso(quando, sizeof(quando));
    if (log) {
      // "desconhecido" segue como rótulo: na importação ele vira observação `unknown`.
      fprintf(log, "e%lu,%s,%s,%s,%.4f\n", static_cast<unsigned long>(contador_eventos),
              quando, arquivo, indice >= 0 ? ROTULOS[indice] : "", prob);
    }
    esp_camera_fb_return(foto);
    vTaskDelay(pdMS_TO_TICKS(INTERVALO_MS));
  }
  if (log) fclose(log);
  delete pre;
  delete modelo;
  esp_camera_deinit();
  esp_vfs_fat_sdcard_unmount("/sdcard", cartao);
}

static void dormir() {
  // ext0 exige os periféricos RTC ligados; pull-down mantém a linha baixa entre disparos.
  esp_sleep_enable_ext0_wakeup(PINO_PIR, 1);
  esp_sleep_pd_config(ESP_PD_DOMAIN_RTC_PERIPH, ESP_PD_OPTION_ON);
  rtc_gpio_pullup_dis(PINO_PIR);
  rtc_gpio_pulldown_en(PINO_PIR);
  esp_deep_sleep_start();
}

extern "C" void app_main(void) {
  if (esp_sleep_get_wakeup_cause() == ESP_SLEEP_WAKEUP_EXT0) {
    rtc_gpio_deinit(PINO_PIR);
    registrar_evento();
  } else {
    ESP_LOGI(TAG, "primeira partida: armando o PIR");
  }
  dormir();
}
