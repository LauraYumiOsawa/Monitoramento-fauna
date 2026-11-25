# Armadilha fotográfica na Arduino Nicla Vision (firmware OpenMV v5) — versão @@ID@@.
#
# Gatilho: PIR em um pino com IRQ (machine.lightsleep acorda por IRQ de pino) ou, sem PIR,
# diferença entre quadros. Classificador TinyML (ShuffleNetV2-x0.5 INT8, quadro inteiro)
# em /rom (ROMFS: o modelo roda direto da flash QSPI). Fotos e registros.csv em /flash
# (11 MB, FAT; a placa não tem microSD). Importe o CSV com
# `python -m animal_monitor_hw import-csv` (ver README).
#
# APIs do OpenMV v5 (csi substitui sensor; ml.Model sem load_to_fb). NÃO EXECUTADO na
# placa. Conferir o nome do pino do PIR e se o GC2145 aceita a resolução pedida.

import csi
import machine
import ml
import os
import time

PINO_PIR = None  # ex.: "PG12" — conferir o pinout; None = gatilho por diferença de quadros
FOTOS, INTERVALO_MS, PAUSA_MS = 3, 300, 10000
LIMIAR_MOVIMENTO = 25  # média da diferença absoluta (0–255) que conta como movimento
PASTA = "/flash"
ROTULOS = ["anta", "tamandua_bandeira", "lobo_guara", "desconhecido"]

cam = csi.CSI()
cam.reset()
cam.pixformat(csi.RGB565)
cam.framesize(csi.QVGA)  # 320 × 240; JPEG por software (o GC2145 não gera JPEG)
cam.snapshot(time=2000)
modelo = ml.Model("/rom/classificador.tflite")


def preparar_log():
    if "registros.csv" not in os.listdir(PASTA):
        with open(PASTA + "/registros.csv", "w") as arquivo:
            arquivo.write("evento,timestamp_iso,arquivo,rotulo,probabilidade\n")


def agora_iso():
    ano, mes, dia, _, hora, minuto, segundo, _ = machine.RTC().datetime()
    return "%04d-%02d-%02dT%02d:%02d:%02d-03:00" % (ano, mes, dia, hora, minuto, segundo)


def classificar(img):
    saida = modelo.predict([img])[0].flatten().tolist()
    melhor = max(range(len(saida)), key=lambda indice: saida[indice])
    # Saída do modelo: logits já desquantizados; softmax para uma probabilidade.
    from math import exp

    maximo = saida[melhor]
    exponenciais = [exp(valor - maximo) for valor in saida]
    return ROTULOS[melhor], exponenciais[melhor] / sum(exponenciais)


def evento(numero):
    with open(PASTA + "/registros.csv", "a") as log:
        for indice in range(FOTOS):
            img = cam.snapshot()
            rotulo, prob = classificar(img)
            nome = "e%d_%d.jpg" % (numero, indice)
            img.save(PASTA + "/" + nome, quality=90)
            log.write("e%d,%s,%s,%s,%.4f\n" % (numero, agora_iso(), nome, rotulo, prob))
            time.sleep_ms(INTERVALO_MS)


def aguardar_movimento(anterior):
    while True:
        atual = cam.snapshot().to_grayscale(copy=True)
        if anterior is not None:
            diferenca = atual.copy().difference(anterior)  # difference() altera a imagem
            if diferenca.get_statistics().mean() > LIMIAR_MOVIMENTO:
                return atual
        anterior = atual
        time.sleep_ms(100)


def main():
    preparar_log()
    numero = len([nome for nome in os.listdir(PASTA) if nome.endswith(".jpg")]) // FOTOS
    if PINO_PIR is not None:
        disparou = [False]
        pir = machine.Pin(PINO_PIR, machine.Pin.IN, machine.Pin.PULL_DOWN)

        def ao_disparar(_pino):
            disparou[0] = True

        pir.irq(ao_disparar, trigger=machine.Pin.IRQ_RISING)
    anterior = None
    while True:
        if PINO_PIR is not None:
            disparou[0] = False
            cam.sleep(True)  # o GC2145 da Nicla não tem pino de power-down ligado:
            machine.lightsleep()  # o consumo em sono fica alto (fórum OpenMV, ~20 mA)
            cam.sleep(False)
            if not disparou[0]:
                continue
        else:
            anterior = aguardar_movimento(anterior)
        numero += 1
        evento(numero)
        time.sleep_ms(PAUSA_MS)


main()
