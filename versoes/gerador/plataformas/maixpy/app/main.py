"""Armadilha fotográfica na MaixCAM — versão @@ID@@ (@@MODELO@@).

PIR (polling: o MaixPy não documenta interrupção de GPIO) → rajada de quadros →
YOLOv8 na NPU (cvimodel INT8) → [Modelo B: classificador no recorte] → JPEG no cartão +
linha em registros.csv, importável para Camtrap DP com
`python -m animal_monitor_hw import-csv` (ver README).

APIs do MaixPy usadas conforme a documentação da Sipeed: nn.YOLOv8 com dual_buff=False
(com True o resultado é do quadro anterior e o recorte sairia do animal errado),
nn.Classifier, camera.Camera, image.crop/save, pinmap + gpio.
"""

import os

from maix import app, camera, display, err, gpio, nn, pinmap, time

PASTA = "/root/armadilha"
MODELO = "@@MODELO@@"  # modelo-a | modelo-b
PINO_PIR = "A19"  # conferir o pinout da sua MaixCAM; IO de 3,3 V (evite A14, A17 e o botão)
LARGURA, ALTURA = 1280, 720  # quadro salvo; o detector redimensiona e remapeia as caixas
QUADROS, INTERVALO_MS, PAUSA_S = 8, 150, 10
MIN_HITS = 3  # Modelo A: quadros com a mesma espécie para confirmar
CONF, IOU = (0.35, 0.45) if MODELO == "modelo-a" else (0.30, 0.40)
MIN_VOTOS, MIN_PARTICIPACAO, MIN_PROB = 3, 0.65, 0.50  # Modelo B: votação da rajada
MARGEM = 0.12
FUSO = "-03:00"
DESCONHECIDO = "desconhecido"


def preparar_pir():
    err.check_raise(pinmap.set_pin_function(PINO_PIR, "GPIO" + PINO_PIR), "pinmap do PIR")
    return gpio.GPIO("GPIO" + PINO_PIR, gpio.Mode.IN, gpio.Pull.PULL_DOWN)


def agora_iso():
    return time.localtime().strftime("%Y-%m-%dT%H:%M:%S") + FUSO


def recorte_quadrado(img, obj):
    """Quadrado centrado na caixa, com margem, dentro da imagem (como no treino)."""
    lado = int(max(obj.w, obj.h) * (1 + 2 * MARGEM))
    lado = max(8, min(lado, img.width(), img.height()))
    cx, cy = obj.x + obj.w // 2, obj.y + obj.h // 2
    x = min(max(0, cx - lado // 2), img.width() - lado)
    y = min(max(0, cy - lado // 2), img.height() - lado)
    return img.crop(x, y, lado, lado)


def analisar_a(deteccoes, rotulos):
    """Porta de MIN_HITS quadros consecutivos por espécie (DetectionGate do Modelo A)."""
    seguidos, melhor = {}, {}
    for objs in deteccoes:
        presentes = set()
        for obj in objs:
            nome = rotulos[obj.class_id]
            presentes.add(nome)
            melhor[nome] = max(melhor.get(nome, 0.0), obj.score)
        for nome in list(seguidos):
            if nome not in presentes:
                seguidos[nome] = 0
        for nome in presentes:
            seguidos[nome] = seguidos.get(nome, 0) + 1
            if seguidos[nome] >= MIN_HITS:
                return "animal", nome, melhor[nome]
    if melhor:
        nome = max(melhor, key=melhor.get)
        return "inconclusivo", nome, melhor[nome]
    return "vazio", "", 0.0


def analisar_b(quadros, deteccoes, classificador):
    """Votação ponderada (escore do detector × probabilidade) sobre o maior animal."""
    votos, contagem, soma_prob = {}, 0, {}
    for img, objs in zip(quadros, deteccoes):
        if not objs:
            continue
        obj = max(objs, key=lambda item: item.score)
        indice, prob = classificador.classify(recorte_quadrado(img, obj))[0]
        nome = classificador.labels[indice]
        votos[nome] = votos.get(nome, 0.0) + obj.score * prob
        soma_prob[nome] = soma_prob.get(nome, 0.0) + prob
        contagem += 1
    if not contagem:
        return "vazio", "", 0.0
    vencedor = max(votos, key=votos.get)
    participacao = votos[vencedor] / sum(votos.values())
    prob_media = soma_prob[vencedor] / contagem
    if (vencedor != DESCONHECIDO and contagem >= MIN_VOTOS and participacao >= MIN_PARTICIPACAO
            and prob_media >= MIN_PROB):
        return "animal", vencedor, participacao
    return "inconclusivo", vencedor, participacao


def registrar(evento, quadros, desfecho, rotulo, prob):
    pasta = f"{PASTA}/midias/{evento}"
    os.makedirs(pasta, exist_ok=True)
    with open(f"{PASTA}/registros.csv", "a") as log:
        for indice, img in enumerate(quadros):
            arquivo = f"midias/{evento}/{indice:03d}.jpg"
            img.save(f"{PASTA}/{arquivo}", quality=90)
            log.write(f"{evento},{agora_iso()},{arquivo},"
                      f"{rotulo if desfecho == 'animal' else desfecho},{prob:.4f}\n")


def main():
    os.makedirs(f"{PASTA}/midias", exist_ok=True)
    if not os.path.exists(f"{PASTA}/registros.csv"):
        with open(f"{PASTA}/registros.csv", "w") as log:
            log.write("evento,timestamp_iso,arquivo,rotulo,probabilidade\n")
    tela = display.Display()
    tela.set_backlight(0)  # sem tela em campo: economiza energia
    detector = nn.YOLOv8(model=f"{PASTA}/modelos/detector.mud", dual_buff=False)
    classificador = None
    if MODELO == "modelo-b":
        classificador = nn.Classifier(model=f"{PASTA}/modelos/classificador.mud", dual_buff=False)
    cam = camera.Camera(LARGURA, ALTURA)
    pir = preparar_pir()
    while not app.need_exit():
        if pir.value() == 0:
            time.sleep_ms(50)
            continue
        evento = f"e{time.time_ms()}"
        quadros, deteccoes = [], []
        for _ in range(QUADROS):
            img = cam.read()
            quadros.append(img)
            deteccoes.append(detector.detect(img, conf_th=CONF, iou_th=IOU))
            time.sleep_ms(INTERVALO_MS)
        if MODELO == "modelo-a":
            desfecho, rotulo, prob = analisar_a(deteccoes, detector.labels)
        else:
            desfecho, rotulo, prob = analisar_b(quadros, deteccoes, classificador)
        # "inconclusivo" fica salvo para revisão; "vazio" também, pois o descarte
        # automático seguro não foi demonstrado (pesquisa, seção 7).
        registrar(evento, quadros, desfecho, rotulo, prob)
        print(evento, desfecho, rotulo, round(prob, 3))
        time.sleep(PAUSA_S)


if __name__ == "__main__":
    main()
