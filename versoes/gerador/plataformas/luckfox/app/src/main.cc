// Armadilha fotográfica — versão @@ID@@: Luckfox Pico (RV1106G3, NPU INT8) com YOLOv8n de
// cabeça dividida (6 saídas box/cls), entrada 320 × 320.
//
// Ciclo: PIR (sysfs GPIO) → rajada de QUADROS → NPU (API C zero-copy do librknnmrt, a única
// disponível no RV1106) → DFL + NMS na CPU → porta de MIN_HITS quadros por espécie →
// JPEGs e registros.csv em /userdata/armadilha, importáveis para Camtrap DP com
// `python -m animal_monitor_hw import-csv`.
//
// Sequência de chamadas RKNN conforme o exemplo yolov8_rv1106_1103.cc do rknn_model_zoo:
// entrada nativa UINT8/NHWC, saídas RKNN_QUERY_NATIVE_NHWC_OUTPUT_ATTR em INT8,
// desquantização (q − zp) · escala. NÃO COMPILADO neste trabalho.

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "rknn_api.h"

namespace {

const char *ROTULOS[] = {"anta", "tamandua_bandeira", "lobo_guara"};
constexpr int NUM_CLASSES = 3;
constexpr int REG_MAX = 16;
constexpr float CONFIANCA = 0.35f;
constexpr float IOU_NMS = 0.45f;
constexpr int QUADROS = 8;
constexpr int MIN_HITS = 3;
constexpr int PAUSA_S = 10;
constexpr int LARGURA_CAMERA = 1280;  // múltiplo de 16 (exigência do opencv-mobile)
constexpr int ALTURA_CAMERA = 720;
// GPIO1_C7 = banco 1 · 32 + grupo C (2) · 8 + 7 = 55 (numeração do wiki da Luckfox).
constexpr int GPIO_PIR = 55;
const char *PASTA = "/userdata/armadilha";

struct Deteccao {
  float x1, y1, x2, y2, escore;
  int classe;
};

struct Letterbox {
  float escala;
  int pad_x, pad_y;
};

bool exportar_gpio(int numero) {
  const std::string base = "/sys/class/gpio/gpio" + std::to_string(numero);
  if (access(base.c_str(), F_OK) != 0) {
    FILE *exportar = fopen("/sys/class/gpio/export", "w");
    if (!exportar) return false;
    fprintf(exportar, "%d", numero);
    fclose(exportar);
  }
  FILE *direcao = fopen((base + "/direction").c_str(), "w");
  if (!direcao) return false;
  fputs("in", direcao);
  fclose(direcao);
  return true;
}

bool pir_ativo(int numero) {
  const std::string caminho = "/sys/class/gpio/gpio" + std::to_string(numero) + "/value";
  char valor = '0';
  const int fd = open(caminho.c_str(), O_RDONLY);
  if (fd < 0) return false;
  const bool ok = read(fd, &valor, 1) == 1;
  close(fd);
  return ok && valor == '1';
}

std::string agora_iso() {
  const std::time_t agora = std::time(nullptr);
  std::tm local{};
  localtime_r(&agora, &local);
  char texto[40];
  std::strftime(texto, sizeof(texto), "%Y-%m-%dT%H:%M:%S%z", &local);
  std::string iso(texto);
  if (iso.size() >= 5) iso.insert(iso.size() - 2, ":");  // -0300 → -03:00
  return iso;
}

class Rede {
 public:
  ~Rede() {
    if (mem_entrada_) rknn_destroy_mem(ctx_, mem_entrada_);
    for (auto *mem : mem_saidas_) rknn_destroy_mem(ctx_, mem);
    if (ctx_) rknn_destroy(ctx_);
  }

  bool abrir(const char *caminho) {
    // tamanho 0: o primeiro argumento é o caminho do .rknn.
    if (rknn_init(&ctx_, const_cast<char *>(caminho), 0, 0, nullptr) != RKNN_SUCC) return false;
    rknn_input_output_num io{};
    if (rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io)) != RKNN_SUCC) return false;
    entrada_.index = 0;
    rknn_query(ctx_, RKNN_QUERY_NATIVE_INPUT_ATTR, &entrada_, sizeof(entrada_));
    // UINT8 funde normalização (mean 0, std 255 da conversão) e quantização na NPU.
    entrada_.type = RKNN_TENSOR_UINT8;
    entrada_.fmt = RKNN_TENSOR_NHWC;
    mem_entrada_ = rknn_create_mem(ctx_, entrada_.size_with_stride);
    if (rknn_set_io_mem(ctx_, mem_entrada_, &entrada_) != RKNN_SUCC) return false;
    saidas_.resize(io.n_output);
    for (uint32_t i = 0; i < io.n_output; i++) {
      saidas_[i].index = i;
      rknn_query(ctx_, RKNN_QUERY_NATIVE_NHWC_OUTPUT_ATTR, &saidas_[i], sizeof(rknn_tensor_attr));
      mem_saidas_.push_back(rknn_create_mem(ctx_, saidas_[i].size_with_stride));
      if (rknn_set_io_mem(ctx_, mem_saidas_[i], &saidas_[i]) != RKNN_SUCC) return false;
    }
    return true;
  }

  int altura() const { return entrada_.dims[1]; }
  int largura() const { return entrada_.dims[2]; }

  Letterbox carregar(const cv::Mat &bgr) {
    const int alt = altura(), larg = largura();
    const float escala = std::min(static_cast<float>(larg) / bgr.cols,
                                  static_cast<float>(alt) / bgr.rows);
    const int nova_larg = static_cast<int>(std::round(bgr.cols * escala));
    const int nova_alt = static_cast<int>(std::round(bgr.rows * escala));
    cv::Mat redimensionada, rgb;
    cv::resize(bgr, redimensionada, cv::Size(nova_larg, nova_alt), 0, 0, cv::INTER_AREA);
    const int pad_x = (larg - nova_larg) / 2, pad_y = (alt - nova_alt) / 2;
    cv::Mat tela(alt, larg, CV_8UC3, cv::Scalar(114, 114, 114));  // mesmo cinza do treino
    redimensionada.copyTo(tela(cv::Rect(pad_x, pad_y, nova_larg, nova_alt)));
    cv::cvtColor(tela, rgb, cv::COLOR_BGR2RGB);
    // A entrada zero-copy exige largura alinhada (w_stride); 320 já é múltiplo de 16.
    const int passo = entrada_.w_stride ? entrada_.w_stride : larg;
    auto *destino = static_cast<uint8_t *>(mem_entrada_->virt_addr);
    for (int y = 0; y < alt; y++) std::memcpy(destino + y * passo * 3, rgb.ptr(y), larg * 3);
    return {escala, pad_x, pad_y};
  }

  bool executar() { return rknn_run(ctx_, nullptr) == RKNN_SUCC; }

  std::vector<Deteccao> decodificar() const {
    // Pares (box 64 canais, cls NUM_CLASSES canais) por grade; cls_sum (1 canal) é ignorado.
    std::vector<Deteccao> candidatas;
    for (size_t i = 0; i < saidas_.size(); i++) {
      const auto &caixa = saidas_[i];
      if (caixa.dims[3] != 4 * REG_MAX) continue;
      for (size_t j = 0; j < saidas_.size(); j++) {
        const auto &cls = saidas_[j];
        if (cls.dims[3] != NUM_CLASSES || cls.dims[1] != caixa.dims[1]) continue;
        decodificar_escala(i, j, candidatas);
      }
    }
    return nms(candidatas);
  }

 private:
  static float desquantizar(int8_t valor, const rknn_tensor_attr &attr) {
    return (static_cast<float>(valor) - attr.zp) * attr.scale;
  }

  void decodificar_escala(size_t i_caixa, size_t i_cls, std::vector<Deteccao> &saida) const {
    const auto &attr_caixa = saidas_[i_caixa];
    const auto &attr_cls = saidas_[i_cls];
    const int grade_h = attr_cls.dims[1], grade_w = attr_cls.dims[2];
    // Passo por célula a partir do tamanho alocado: cobre preenchimento de canais, se houver.
    const int passo_cls = attr_cls.size_with_stride / (grade_h * grade_w);
    const int passo_caixa = attr_caixa.size_with_stride / (grade_h * grade_w);
    const auto *dados_cls = static_cast<const int8_t *>(mem_saidas_[i_cls]->virt_addr);
    const auto *dados_caixa = static_cast<const int8_t *>(mem_saidas_[i_caixa]->virt_addr);
    const float passo_x = static_cast<float>(largura()) / grade_w;
    const float passo_y = static_cast<float>(altura()) / grade_h;
    for (int y = 0; y < grade_h; y++) {
      for (int x = 0; x < grade_w; x++) {
        const int celula = y * grade_w + x;
        int melhor = 0;
        float escore = -1.0f;
        for (int c = 0; c < NUM_CLASSES; c++) {
          const float valor = desquantizar(dados_cls[celula * passo_cls + c], attr_cls);
          if (valor > escore) {
            escore = valor;
            melhor = c;
          }
        }
        if (escore < CONFIANCA) continue;  // filtra antes da DFL: quase tudo é fundo
        float distancia[4];
        for (int lado = 0; lado < 4; lado++) {
          float logits[REG_MAX], maximo = -1e9f, soma = 0.0f, esperanca = 0.0f;
          for (int k = 0; k < REG_MAX; k++) {
            logits[k] = desquantizar(dados_caixa[celula * passo_caixa + lado * REG_MAX + k],
                                     attr_caixa);
            maximo = std::max(maximo, logits[k]);
          }
          for (int k = 0; k < REG_MAX; k++) soma += (logits[k] = std::exp(logits[k] - maximo));
          for (int k = 0; k < REG_MAX; k++) esperanca += k * logits[k] / soma;
          distancia[lado] = esperanca;
        }
        const float cx = x + 0.5f, cy = y + 0.5f;
        saida.push_back({(cx - distancia[0]) * passo_x, (cy - distancia[1]) * passo_y,
                         (cx + distancia[2]) * passo_x, (cy + distancia[3]) * passo_y, escore,
                         melhor});
      }
    }
  }

  static float iou(const Deteccao &a, const Deteccao &b) {
    const float x1 = std::max(a.x1, b.x1), y1 = std::max(a.y1, b.y1);
    const float x2 = std::min(a.x2, b.x2), y2 = std::min(a.y2, b.y2);
    const float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    const float uniao = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
    return uniao > 0 ? inter / uniao : 0.0f;
  }

  static std::vector<Deteccao> nms(std::vector<Deteccao> candidatas) {
    std::sort(candidatas.begin(), candidatas.end(),
              [](const Deteccao &a, const Deteccao &b) { return a.escore > b.escore; });
    std::vector<Deteccao> mantidas;
    for (const auto &candidata : candidatas) {
      bool suprimida = false;
      for (const auto &mantida : mantidas) {
        if (mantida.classe == candidata.classe && iou(mantida, candidata) > IOU_NMS) {
          suprimida = true;
          break;
        }
      }
      if (!suprimida) mantidas.push_back(candidata);
    }
    return mantidas;
  }

  rknn_context ctx_ = 0;
  rknn_tensor_attr entrada_{};
  rknn_tensor_mem *mem_entrada_ = nullptr;
  std::vector<rknn_tensor_attr> saidas_;
  std::vector<rknn_tensor_mem *> mem_saidas_;
};

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "uso: %s modelos/a_320_split_rv1106.rknn\n", argv[0]);
    return 2;
  }
  Rede rede;
  if (!rede.abrir(argv[1])) {
    std::fprintf(stderr, "falha ao carregar %s\n", argv[1]);
    return 1;
  }
  cv::VideoCapture camera;
  camera.set(cv::CAP_PROP_FRAME_WIDTH, LARGURA_CAMERA);
  camera.set(cv::CAP_PROP_FRAME_HEIGHT, ALTURA_CAMERA);
  camera.open(0);
  if (!camera.isOpened()) {
    std::fprintf(stderr, "câmera indisponível (rode RkLunch-stop.sh antes)\n");
    return 1;
  }
  if (!exportar_gpio(GPIO_PIR)) std::fprintf(stderr, "aviso: GPIO %d indisponível\n", GPIO_PIR);
  mkdir(PASTA, 0775);
  const std::string midias = std::string(PASTA) + "/midias";
  mkdir(midias.c_str(), 0775);
  const std::string log = std::string(PASTA) + "/registros.csv";
  if (access(log.c_str(), F_OK) != 0) {
    FILE *novo = std::fopen(log.c_str(), "w");
    std::fputs("evento,timestamp_iso,arquivo,rotulo,probabilidade\n", novo);
    std::fclose(novo);
  }

  while (true) {
    if (!pir_ativo(GPIO_PIR)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      continue;
    }
    const std::string evento = "e" + std::to_string(std::time(nullptr));
    int seguidos[NUM_CLASSES] = {0};
    float melhor[NUM_CLASSES] = {0};
    int confirmado = -1;
    bool houve = false;
    std::vector<std::string> arquivos;
    for (int q = 0; q < QUADROS; q++) {
      cv::Mat quadro;
      camera >> quadro;
      if (quadro.empty()) continue;
      rede.carregar(quadro);
      if (!rede.executar()) continue;
      bool presente[NUM_CLASSES] = {false};
      for (const auto &d : rede.decodificar()) {
        presente[d.classe] = true;
        melhor[d.classe] = std::max(melhor[d.classe], d.escore);
        houve = true;
      }
      for (int c = 0; c < NUM_CLASSES; c++) {
        seguidos[c] = presente[c] ? seguidos[c] + 1 : 0;
        if (confirmado < 0 && seguidos[c] >= MIN_HITS) confirmado = c;
      }
      const std::string nome = "midias/" + evento + "_" + std::to_string(q) + ".jpg";
      if (cv::imwrite(std::string(PASTA) + "/" + nome, quadro)) arquivos.push_back(nome);
    }
    // "inconclusivo" (detecção sem confirmação) e "vazio" ficam gravados para revisão.
    const std::string rotulo =
        confirmado >= 0 ? ROTULOS[confirmado] : (houve ? "inconclusivo" : "vazio");
    const float prob = confirmado >= 0 ? melhor[confirmado] : 0.0f;
    FILE *registro = std::fopen(log.c_str(), "a");
    for (const auto &nome : arquivos) {
      std::fprintf(registro, "%s,%s,%s,%s,%.4f\n", evento.c_str(), agora_iso().c_str(),
                   nome.c_str(), rotulo.c_str(), prob);
    }
    std::fclose(registro);
    std::printf("%s: %s (%.2f), %zu imagens\n", evento.c_str(), rotulo.c_str(), prob,
                arquivos.size());
    std::this_thread::sleep_for(std::chrono::seconds(PAUSA_S));
  }
}
