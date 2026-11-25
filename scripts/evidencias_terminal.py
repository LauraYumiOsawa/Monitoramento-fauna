#!/usr/bin/env python3
"""Gera prints de terminal (comando + saída real) para o relatório do TCC.

Cada sessão é executada de verdade por um bash dentro de um pseudo-terminal (PTY), com o
.venv do projeto ativado, na raiz do repositório. A transcrição capturada (prompt,
comandos e saída, inclusive as cores) é gravada em ``outputs/evidencias/terminal/`` como
``TNN_nome.log`` (texto puro) e ``TNN_nome.png`` (janela de terminal renderizada pelo
Chrome headless). Nada é editado na saída: o que aparece no PNG é o que o bash imprimiu.

    .venv/bin/python scripts/evidencias_terminal.py --listar
    .venv/bin/python scripts/evidencias_terminal.py              # todas (~20 min)
    .venv/bin/python scripts/evidencias_terminal.py T02 T06      # só algumas
    .venv/bin/python scripts/evidencias_terminal.py --so-png     # re-renderiza dos .log
"""

from __future__ import annotations

import argparse
import fcntl
import getpass
import html
import json
import os
import re
import shutil
import socket
import struct
import subprocess
import sys
import termios
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAIDA = ROOT / "outputs" / "evidencias" / "terminal"
COLUNAS = 128
VENV = ROOT / ".venv"
USUARIO = getpass.getuser()
HOST = socket.gethostname()
# Bash com PROMPT_DIRTRIM=1 mostra só o último diretório depois de "~/.../".
DIR_PROMPT = "~/.../" + ROOT.name
DIR_TITULO = "~/" + str(ROOT.relative_to(Path.home())) if ROOT.is_relative_to(Path.home()) else str(ROOT)
MARCA = "\x1eFIM{}\x1e"
T = "outputs/evidencias/terminal"  # atalho usado nos comandos


@dataclass
class Sessao:
    codigo: str
    nome: str
    titulo: str
    comandos: list[str]


def c(texto: str) -> str:
    return textwrap.dedent(texto).strip("\n")


DOCKER_ARM64 = c(r"""
    docker run --rm --platform linux/arm64 --network none --read-only --tmpfs /tmp \
      --user "$(id -u):$(id -g)" -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 \
      -e PYTHONPATH=/work/artefatos/src {env} \
      --cpus {cpus} --memory {mem} --memory-swap {mem} \
      -v "$PWD/../../avaliacao_embarcada":/work:ro \
      -v "$PWD/../../avaliacao_embarcada/qemu-aarch64":/qemu-aarch64:ro \
      --entrypoint /qemu-aarch64 animal-monitor:arm64 -cpu {cpu} \
      /usr/local/bin/python python /work/{script} | jq -c '{filtro}'
""")


def docker_arm64(cpu: str, cpus: str, mem: str, script: str, env: str, filtro: str) -> str:
    return DOCKER_ARM64.format(cpu=cpu, cpus=cpus, mem=mem, script=script, env=env, filtro=filtro)


SMOKE = "{model, architecture, input_shape, output_shapes, finite}"
PIPE = "{variant, image, predictions}"

SESSOES = [
    Sessao("T01", "ambiente", "Ambiente de execução", [
        "date",
        "lscpu | grep -E '^(Model name|CPU\\(s\\)):'",
        "free -h | head -n 2",
        "python --version && animal-monitor info | jq -c .packages",
        "docker --version && docker image ls animal-monitor",
        "../../avaliacao_embarcada/qemu-aarch64 --version | head -n 1",
        c("""
        jq -c '.host | {onnxruntime, cpus_logicas: (.affinity | length)}' \\
          ../../avaliacao_embarcada/resultados_cpu_host.json
        """),
    ]),
    Sessao("T02", "dataset", "Validação do conjunto de dados", [
        "date",
        c(f"""
        animal-monitor validate-dataset \\
          --data config/dataset.yaml \\
          --json-output {T}/dataset.json | tail -n 1
        """),
        c(f"""
        jq -c '.splits | to_entries[] | {{split: .key,
          imagens: .value.images, caixas: .value.boxes, por_classe: .value.class_boxes}}' \\
          {T}/dataset.json
        """),
        c(f"""
        jq -c '{{imagens: ([.splits[].images] | add),
          caixas: ([.splits[].boxes] | add), problemas: (.issues | length)}}' {T}/dataset.json
        """),
        c("""
        for s in train val test; do
          echo "$s: $(ls data/animals/images/$s | grep -c '^Gemini_Generated_Image_.*png$') imagens geradas por IA"
        done
        """),
    ]),
    Sessao("T03", "testes_x86", "Testes automatizados em x86_64", [
        "date",
        c("""
        python -m unittest tests.test_config tests.test_dataset \\
          tests.test_events tests.test_prepost tests.test_splitting
        """),
        c(f"""
        python scripts/testar_tudo.py \\
          --output-dir {T}/testes_x86 2>&1 | grep -E '^(Ran|OK|FAILED)'
        """),
        f"jq -c '{{tests, compile_ok, platform}}' {T}/testes_x86/resultado.json",
    ]),
    Sessao("T04", "testes_arm64", "Testes automatizados em ARM64 emulado (QEMU, Cortex-A76)", [
        "date",
        f"mkdir -p {T}/testes_arm64",
        c(f"""
        docker run --rm --platform linux/arm64 --network none --read-only --tmpfs /tmp \\
          --user "$(id -u):$(id -g)" -e HOME=/tmp -e PYTHONPYCACHEPREFIX=/tmp/pyc \\
          -v "$PWD":/work:ro -v "$PWD/{T}/testes_arm64":/out \\
          -v "$PWD/../../avaliacao_embarcada/qemu-aarch64":/qemu-aarch64:ro \\
          --entrypoint /qemu-aarch64 animal-monitor:arm64 -cpu cortex-a76 \\
          /usr/local/bin/python python /work/scripts/testar_tudo.py --output-dir /out \\
          2>&1 | grep -E '^(Ran|OK|FAILED)'
        """),
        f"jq -c '{{tests, compile_ok, platform, python}}' {T}/testes_arm64/resultado.json",
    ]),
    Sessao("T05", "integridade", "Integridade SHA-256 dos artefatos ONNX", [
        "date",
        c(f"""
        jq -r '.variants[].artifacts[] | "\\(.sha256)  models/\\(.path)"' \\
          ../../avaliacao_embarcada/resultados_cpu_host.json | sort -u > {T}/sha256_11-11.txt
        """),
        f"sha256sum -c {T}/sha256_11-11.txt",
        c("""
        for f in models/model_metadata.json models/modelo_b/*_metadata.json; do
          jq -r --arg d "$(dirname "$f")" '"\\(.sha256)  \\($d)/\\(.model_file)"' "$f"
        done | sha256sum -c -
        """),
        "cd models/modelo_b_mobilenet_v3_small && sha256sum species_classifier_int8*.onnx; cd - > /dev/null",
    ]),
    Sessao("T06", "quadro3", "Reavaliação das 10 configurações do Quadro 3 (CLI animal-monitor-b)", [
        "date",
        c(f"""
        cat > {T}/quadro3.tsv <<'EOF'
        B_V2_FP32 modelo_b/presence_detector.onnx modelo_b/species_classifier.onnx
        B_V2_INT8 modelo_b/presence_detector_int8.onnx modelo_b/species_classifier_int8.onnx
        B_V3_FP32 modelo_b/presence_detector.onnx modelo_b_mobilenet_v3_small/species_classifier.onnx
        B_V3_INT8_MinMax modelo_b/presence_detector_int8.onnx modelo_b_mobilenet_v3_small/species_classifier_int8.onnx
        B_V3_INT8_Pctl modelo_b/presence_detector_int8.onnx modelo_b_mobilenet_v3_small/species_classifier_int8_percentile.onnx
        B_V3_INT8_Entropy modelo_b/presence_detector_int8.onnx modelo_b_mobilenet_v3_small/species_classifier_int8_entropy.onnx
        B_V3_INT8_Conv modelo_b/presence_detector_int8.onnx modelo_b_mobilenet_v3_small/species_classifier_int8_conv.onnx
        B_Shuffle_FP32 modelo_b/presence_detector.onnx modelo_b_shufflenet_v2_x0_5/species_classifier.onnx
        B_Shuffle_INT8 modelo_b/presence_detector_int8.onnx modelo_b_shufflenet_v2_x0_5/species_classifier_int8.onnx
        EOF
        """),
        c(f"""
        cat > {T}/linha.jq <<'EOF'
        [.model, (.map50, ."precision@0.35", ."recall@0.35", ."f1@0.35" | . * 1000 | round / 1000),
         .counts.true_positive, .counts.false_positive, .counts.false_negative, $mib] | @tsv
        EOF
        """),
        c(f"""
        {{ echo "configuração mAP50 precisão recall F1 VP FP FN MiB"
          mib=$(du -cb models/animal_detector.onnx | tail -n 1 | awk '{{printf "%.2f", $1/1048576}}')
          animal-monitor-b evaluate-baseline --model models/animal_detector.onnx \\
            --label A_YOLOv8n_FP32 --output {T}/q3_A.json | jq -r --arg mib $mib -f {T}/linha.jq
          while read nome det cls; do
            mib=$(du -cb models/$det models/$cls | tail -n 1 | awk '{{printf "%.2f", $1/1048576}}')
            animal-monitor-b evaluate --detector models/$det --classifier models/$cls \\
              --label $nome --output {T}/q3_$nome.json | jq -r --arg mib $mib -f {T}/linha.jq
          done < {T}/quadro3.tsv
        }} | column -t
        """),
    ]),
    Sessao("T07", "modelo_b_detalhe", "Modelo B (MobileNetV2 FP32): AP por espécie, contagens e confusão", [
        f"jq -c '.ap50_by_class | map_values(.ap50 * 1000 | round / 1000)' {T}/q3_B_V2_FP32.json",
        f"jq -c '.counts' {T}/q3_B_V2_FP32.json",
        f"jq -c '.species_confusion | {{linhas: .rows, colunas: .columns}}' {T}/q3_B_V2_FP32.json",
        f"jq -c '.species_confusion.matrix[]' {T}/q3_B_V2_FP32.json",
        f"jq -c '{{A: .counts}}' {T}/q3_A.json; jq -c '{{B_INT8: .counts}}' {T}/q3_B_V2_INT8.json",
    ]),
    Sessao("T08", "classificadores", "Classificadores nos 46 recortes de teste (treino e reexecução ONNX)", [
        "date",
        "for d in data/crops/test/*/; do echo \"$(ls \"$d\" | wc -l) recortes em $d\"; done",
        c("""
        jq -r '"\\(.backbone)\\tteste PyTorch: \\(.test_report.accuracy)"' \\
          models/modelo_b/classifier_training.json \\
          models/modelo_b_mobilenet_v3_small/classifier_training.json \\
          models/modelo_b_shufflenet_v2_x0_5/classifier_training.json
        """),
        c("""
        python - <<'EOF'
        import json, numpy as np, onnxruntime as ort
        from pathlib import Path
        from animal_monitor_b.augment import resize_square, to_tensor
        from animal_monitor_b.classifier import CropCache
        for pasta in ["modelo_b", "modelo_b_mobilenet_v3_small", "modelo_b_shufflenet_v2_x0_5"]:
            for onnx in sorted(Path("models", pasta).glob("species_classifier*.onnx")):
                rotulos = json.load(open(onnx.parent / "classifier_metadata.json"))["labels"]
                recortes = CropCache(Path("data/crops"), "test", rotulos)
                sessao = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
                entrada = sessao.get_inputs()[0].name
                acertos = sum(int(np.argmax(sessao.run(None, {entrada: to_tensor(resize_square(img, 128))[None]})[0])) == alvo
                              for img, alvo in zip(recortes.images, recortes.targets))
                print(f"{str(onnx):70} {acertos}/{len(recortes)} = {100 * acertos / len(recortes):.2f}%")
        EOF
        """),
    ]),
    Sessao("T09", "quantizacao_detector", "Detector de presença: INT8 só convoluções × todos os operadores", [
        c("""
        python - 2>/dev/null <<'EOF'
        import cv2, sys
        from pathlib import Path
        sys.path[:0] = ["versoes/comum"]
        from animal_monitor_hw.backends.onnxruntime_backend import OnnxRuntimeRunner
        from animal_monitor_hw.models import YoloDetector
        from animal_monitor_b.evaluation import _split_images
        imagens = [cv2.imread(str(p)) for p in _split_images(Path("config/dataset.yaml"), "test")]
        for onnx in ["presence_detector.onnx", "presence_detector_int8.onnx", "presence_detector_int8_todos_ops.onnx"]:
            caminho = f"models/modelo_b/{onnx}"
            det = YoloDetector(OnnxRuntimeRunner(caminho), {0: "animal"}, 0.30, 0.40, output_format="standard")
            escores = [float(s) for img in imagens for s in det.detect_all(img)[1]]
            print(f"{onnx:40} caixas={len(escores):2}  escore médio={sum(escores)/len(escores):.3f}  máximo={max(escores):.3f}")
        EOF
        """),
        "jq -c '.robustez_int8_detector_18_imagens_teste' outputs/modelo_b/resumo_comparativo.json",
    ]),
    Sessao("T10", "resolucao", "Resolução de entrada do Modelo A e saída YOLO dividida (medir_proxies.py)", [
        "date",
        c("""
        PYTHONPATH=versoes/comum/ferramentas python - 2>/dev/null <<'EOF'
        import medir_proxies as m
        for nome in ["a_640", "a_512", "a_320", "a_224x320", "a_256", "a_224", "a_192"]:
            r = m.resumo(m.avaliar_a(m.REFERENCIA.get(nome, m.MODELOS / f"{nome}.onnx"), "standard"))
            print(f"{nome:10} mAP50={r['map50']:.3f}  P={r['precisao']:.3f}  R={r['recall']:.3f}  F1={r['f1']:.3f}")
        nomes = m.load_class_names(m.DATASET)
        print("A 640:", m.equivalencia_dividida(m.REFERENCIA["a_640"], m.MODELOS / "a_640_split.onnx", nomes))
        print("B 416:", m.equivalencia_dividida(m.REFERENCIA["b_416"], m.MODELOS / "b_416_split.onnx", {0: "animal"}))
        EOF
        """),
    ]),
    Sessao("T11", "rajadas", "Simulação de rajadas (simular_rajadas.py) comparada ao JSON salvo", [
        "date",
        c("""
        PYTHONPATH=versoes/comum/ferramentas python - 2>/dev/null <<'EOF'
        import json, simular_rajadas as s
        salvo = json.load(open("versoes/ranking/medicoes/rajadas_x86.json"))["proxies"]
        for proxy in s.PROXIES:
            r = s.simular(proxy)
            d = r["desfechos"]
            print(f"{proxy:31} eventos={r['eventos']} correto={d.get('correto', 0):2} errado={d.get('errado', 0):2} "
                  f"inconclusivo={d.get('inconclusivo', 0):2} vazio={d.get('vazio', 0):2} igual_ao_salvo={d == salvo[proxy]['desfechos']}")
        EOF
        """),
    ]),
    Sessao("T12", "videos_modelo_a", "Eventos do Modelo A nos três vídeos (animal-monitor infer)", [
        "date",
        c(f"""
        sed -e 's#^output_dir:.*#output_dir: videos_a#' \\
            -e 's#^database:.*#database: videos_a/eventos.sqlite3#' \\
            config/runtime.yaml > {T}/runtime_a.yaml
        """),
        c(f"""
        for v in anta lobo_guara tamandua_bandeira; do
          rm -f {T}/videos_a/eventos.sqlite3
          animal-monitor infer --model models/animal_detector.onnx \\
            --source samples/video_$v.mp4 --runtime {T}/runtime_a.yaml \\
            | jq -c '{{source, processed_frames, detections, new_events}}'
        done
        """),
    ]),
    Sessao("T13", "videos_modelo_b", "Eventos do Modelo B nos vídeos com NMS 0,40 e 0,50 (animal-monitor-b)", [
        "date",
        c(f"""
        for iou in 0.40 0.50; do
          sed -e "s#^detector_iou:.*#detector_iou: $iou#" \\
              -e "s#^output_dir:.*#output_dir: videos_b_$iou#" \\
              -e "s#^database:.*#database: videos_b_$iou/eventos.sqlite3#" \\
              config/modelo_b/runtime.yaml > {T}/runtime_b_$iou.yaml
        done
        """),
        c(f"""
        for iou in 0.40 0.50; do
          for v in anta lobo_guara tamandua_bandeira; do
            rm -f {T}/videos_b_$iou/eventos.sqlite3
            animal-monitor-b infer --detector models/modelo_b/presence_detector.onnx \\
              --classifier models/modelo_b/species_classifier.onnx \\
              --source samples/video_$v.mp4 --runtime {T}/runtime_b_$iou.yaml \\
              | jq -c --arg iou $iou '{{nms: $iou, source, tracks_created, new_sightings}}'
          done
        done
        """),
        c(f"""
        for iou in 0.40 0.50; do
          animal-monitor-b evaluate --detector models/modelo_b/presence_detector.onnx \\
            --classifier models/modelo_b/species_classifier.onnx \\
            --runtime {T}/runtime_b_$iou.yaml --output {T}/nms_$iou.json \\
            | jq -c --arg iou $iou '{{nms: $iou, map50, precisao: ."precision@0.35", recall: ."recall@0.35"}}'
        done
        """),
    ]),
    Sessao("T14", "speciesnet", "MegaDetector v5a + SpeciesNet (avaliar_speciesnet.py sobre as predições salvas)", [
        "date",
        c(f"""
        python versoes/comum/ferramentas/avaliar_speciesnet.py \\
          versoes/ranking/medicoes/speciesnet_predicoes_test.json \\
          --saida {T}/speciesnet.json
        """),
        c("""
        jq '[.predictions[].detections[]? | select(.category == "1" and .conf >= 0.2)] | length' \\
          versoes/ranking/medicoes/speciesnet_predicoes_test.json
        """),
    ]),
    Sessao("T15", "benchmark_quadro4", "Benchmark em CPU: valores do Quadro 4 e reexecução (avaliar_modelos.py)", [
        "date",
        c("""
        jq -r '.variants | to_entries[] | select(.key | test("YOLOv8n|MobileNetV2"))
          | [.key, (.value.bench_4threads | .mean_ms, .p50_ms, .p95_ms, .throughput_sequential_fps
          | . * 100 | round / 100)] | @tsv' ../../avaliacao_embarcada/resultados_cpu_host.json | column -t
        """),
        c("""
        PYTHONPATH=../../avaliacao_embarcada python - 2>/dev/null <<'EOF'
        import cv2, avaliar_modelos as m
        imagens = [cv2.imread(str(p)) for p in sorted((m.ART / "data/animals/images/test").iterdir())]
        for nome in ["A_YOLOv8n_FP32", "B_MobileNetV2_FP32", "B_MobileNetV2_INT8"]:
            preditor, _ = m.make_predictor(*m.VARIANTS[nome], 4)
            b = m.benchmark(preditor, imagens)
            print(f"{nome:20} média={b['mean_ms']:6.2f} ms  p50={b['p50_ms']:6.2f} ms  "
                  f"p95={b['p95_ms']:6.2f} ms  {b['throughput_sequential_fps']:5.2f} img/s  ({b['calls']} chamadas)")
        EOF
        """),
    ]),
    Sessao("T16", "fps_videos", "Velocidade nos vídeos: médias dos benchmarks salvos", [
        c("""
        for m in modelo_a_4t modelo_b_fp32_4t modelo_b_int8_4t; do
          jq -s -c --arg m $m '{modelo: $m, frames: (map(.measured_frames) | add),
            fps_medio: (map(.throughput_fps) | add / length),
            latencia_media_ms: (map(.latency_ms.total.mean // .latency_ms.mean) | add / length)}' \\
            outputs/modelo_b/benchmark_video_*_$m.json
        done
        """),
        "jq -c .condicoes outputs/modelo_b/resumo_comparativo.json",
        c("""
        jq -s -c '{sim_embedded_fps: map(.throughput_fps), frames: map(.measured_frames),
          media: (map(.throughput_fps) | add / length)}' outputs/sim_embedded/benchmark_video_*.json
        """),
    ]),
    Sessao("T17", "arm64_pi5", "ARM64 · perfil Raspberry Pi 5 (Cortex-A76, 4 CPUs, 4 GiB)", [
        "date",
        docker_arm64("cortex-a76", "4", "4096m", "arm64_smoke.py",
                     "-e TCC_MODELS=animal_detector.onnx,modelo_b/presence_detector.onnx,modelo_b/species_classifier.onnx",
                     SMOKE),
        "for v in A B; do\n" + textwrap.indent(
            docker_arm64("cortex-a76", "4", "4096m", "arm64_pipeline.py", "-e TCC_VARIANT=$v", PIPE), "  ")
        + "\ndone",
    ]),
    Sessao("T18", "arm64_pizero", "ARM64 · perfil Raspberry Pi Zero 2 W (Cortex-A53, limite de 512 MiB)", [
        "date",
        docker_arm64("cortex-a53", "4", "512m", "arm64_smoke.py",
                     "-e TCC_MODELS=modelo_b/presence_detector_int8.onnx,modelo_b/species_classifier_int8.onnx",
                     SMOKE),
        docker_arm64("cortex-a53", "4", "512m", "arm64_pipeline.py", "-e TCC_VARIANT=B_INT8", PIPE),
    ]),
    Sessao("T19", "arm64_opi5_jetson", "ARM64 · perfis Orange Pi 5 (A76, 4 GiB) e Jetson Orin Nano Super (A78AE, 8 GiB)", [
        "date",
        docker_arm64("cortex-a76", "4", "4096m", "arm64_smoke.py",
                     "-e TCC_MODELS=modelo_b/presence_detector_int8.onnx,modelo_b/species_classifier_int8.onnx",
                     SMOKE),
        docker_arm64("cortex-a76", "4", "4096m", "arm64_pipeline.py", "-e TCC_VARIANT=B_INT8", PIPE),
        docker_arm64("cortex-a78ae", "6", "8192m", "arm64_smoke.py",
                     "-e TCC_MODELS=modelo_b/presence_detector.onnx,modelo_b/species_classifier.onnx",
                     SMOKE),
        docker_arm64("cortex-a78ae", "6", "8192m", "arm64_pipeline.py", "-e TCC_VARIANT=B", PIPE),
    ]),
    Sessao("T20", "cruzada_x86", "Verificação cruzada: mesma imagem no x86_64 (host) e valores ARM64 salvos", [
        "uname -m",
        c("""
        for v in A B B_INT8; do
          TCC_VARIANT=$v python ../../avaliacao_embarcada/arm64_pipeline.py | jq -c '{variant, predictions}'
        done
        """),
        c("""
        jq -r '.profiles[] | .profile as $p | .real_image_pipeline[]
          | "\\($p): \\(.variant) \\(.class) \\(.confidence) \\(.box_xyxy)"' \\
          ../../avaliacao_embarcada/resultados_arm64_qemu.json
        """),
    ]),
    Sessao("T21", "matriz", "Matriz das 25 versões: QEMU ARM64 e Docker com limites (testar_matriz.py)", [
        "date",
        f"python scripts/testar_matriz.py --output-dir {T}/matriz",
    ]),
]


# --------------------------------------------------------------------------- execução


def ambiente() -> dict[str, str]:
    env = dict(os.environ)
    for chave in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
        env.pop(chave, None)
    env.update({
        "PATH": f"{VENV / 'bin'}:{env.get('PATH', '')}",
        "VIRTUAL_ENV": str(VENV),
        "TERM": "xterm-256color",
        "COLUMNS": str(COLUNAS),
        "LINES": "60",
    })
    return env


def executar(sessao: Sessao) -> list[tuple[str, str]]:
    roteiro = ["set +e", f"cd '{ROOT}'"]
    for indice, comando in enumerate(sessao.comandos):
        roteiro += [comando, f"printf '\\n{MARCA.format(indice)}\\n'"]
    arquivo = SAIDA / f".{sessao.codigo}.sh"
    arquivo.write_text("\n".join(roteiro) + "\n", encoding="utf-8")

    mestre, escravo = os.openpty()
    fcntl.ioctl(escravo, termios.TIOCSWINSZ, struct.pack("HHHH", 60, COLUNAS, 0, 0))
    processo = subprocess.Popen(["bash", "--norc", "--noprofile", str(arquivo)], stdin=escravo,
                                stdout=escravo, stderr=escravo, cwd=ROOT, env=ambiente(),
                                start_new_session=True)
    os.close(escravo)
    dados = bytearray()
    while True:
        try:
            bloco = os.read(mestre, 65536)
        except OSError:
            break
        if not bloco:
            break
        dados += bloco
    processo.wait()
    os.close(mestre)
    arquivo.unlink(missing_ok=True)

    texto = dados.decode("utf-8", errors="replace").replace("\r\n", "\n")
    texto = "\n".join(linha.rsplit("\r", 1)[-1] for linha in texto.split("\n"))
    resultado = []
    for indice, comando in enumerate(sessao.comandos):
        antes, _, texto = texto.partition(MARCA.format(indice))
        resultado.append((comando, antes.strip("\n")))
    return resultado


# --------------------------------------------------------------------------- transcrição


def prompt_texto() -> str:
    return f"(.venv) {USUARIO}@{HOST}:{DIR_PROMPT}$ "


ANSI = re.compile(r"\x1b\[([0-9;?]*)([A-Za-z])|\x1b\][^\x07]*\x07|\x1b[()][0-9A-Za-z]")


def sem_ansi(texto: str) -> str:
    return ANSI.sub("", texto)


def gravar_log(sessao: Sessao, passos: list[tuple[str, str]]) -> Path:
    linhas = []
    for comando, saida in passos:
        primeira, *resto = comando.split("\n")
        linhas.append(prompt_texto() + primeira)
        linhas += [f"> {linha}" for linha in resto]
        if saida:
            linhas += sem_ansi(saida).split("\n")
    linhas.append(prompt_texto())
    destino = SAIDA / f"{sessao.codigo}_{sessao.nome}.log"
    destino.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    (SAIDA / f"{sessao.codigo}_{sessao.nome}.json").write_text(
        json.dumps({"titulo": sessao.titulo, "passos": passos}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    return destino


# --------------------------------------------------------------------------- renderização

PALETA = ["#2e3436", "#cc0000", "#4e9a06", "#c4a000", "#3465a4", "#75507b", "#06989a", "#d3d7cf",
          "#555753", "#ef2929", "#8ae234", "#fce94f", "#729fcf", "#ad7fa8", "#34e2e2", "#eeeeec"]
FUNDO, FRENTE = "#1e1e1e", "#d0cfcc"


def ansi_para_html(texto: str) -> list[str]:
    """Converte texto com SGR em linhas HTML, quebrando em COLUNAS como o terminal faz."""
    linhas: list[str] = []
    estado = {"cor": None, "negrito": False}
    for linha in texto.split("\n"):
        pedacos: list[tuple[str, dict]] = []
        pos = 0
        for m in ANSI.finditer(linha):
            if m.start() > pos:
                pedacos.append((linha[pos:m.start()], dict(estado)))
            pos = m.end()
            if m.group(2) == "m":
                codigos = [int(x) if x.isdigit() else 0 for x in (m.group(1) or "0").split(";")]
                i = 0
                while i < len(codigos):
                    cod = codigos[i]
                    if cod == 0:
                        estado = {"cor": None, "negrito": False}
                    elif cod == 1:
                        estado["negrito"] = True
                    elif cod == 22:
                        estado["negrito"] = False
                    elif 30 <= cod <= 37:
                        estado["cor"] = cod - 30
                    elif 90 <= cod <= 97:
                        estado["cor"] = cod - 90 + 8
                    elif cod == 39:
                        estado["cor"] = None
                    elif cod == 38 and i + 2 < len(codigos) and codigos[i + 1] == 5:
                        estado["cor"] = codigos[i + 2] if codigos[i + 2] < 16 else 7
                        i += 2
                    i += 1
        if pos < len(linha):
            pedacos.append((linha[pos:], dict(estado)))
        # quebra visual em COLUNAS
        atual, largura = [], 0
        for trecho, est in pedacos or [("", estado)]:
            while trecho:
                cabe = COLUNAS - largura
                parte, trecho = trecho[:cabe], trecho[cabe:]
                atual.append(_span(parte, est))
                largura += len(parte)
                if largura >= COLUNAS and trecho:
                    linhas.append("".join(atual))
                    atual, largura = [], 0
        linhas.append("".join(atual))
    return linhas


def _span(texto: str, est: dict) -> str:
    texto = html.escape(texto)
    estilos = []
    if est["cor"] is not None:
        cor = est["cor"] + (8 if est["negrito"] and est["cor"] < 8 else 0)
        estilos.append(f"color:{PALETA[cor]}")
    elif est["negrito"]:
        estilos.append("color:#ffffff")
    if est["negrito"]:
        estilos.append("font-weight:bold")
    return f'<span style="{";".join(estilos)}">{texto}</span>' if estilos else texto


def prompt_html() -> str:
    return ("(.venv) " + f'<span class="u">{html.escape(f"{USUARIO}@{HOST}")}</span>:'
            + f'<span class="d">{html.escape(DIR_PROMPT)}</span>$ ')


def linhas_comando(comando: str) -> list[str]:
    """Prompt + comando como digitado; linhas seguintes com o prompt de continuação '> '."""
    resultado = []
    for indice, linha in enumerate(comando.split("\n")):
        prefixo_txt = prompt_texto() if indice == 0 else "> "
        prefixo = prompt_html() if indice == 0 else "> "
        cheio = prefixo_txt + linha
        partes = [cheio[i:i + COLUNAS] for i in range(0, max(len(cheio), 1), COLUNAS)]
        primeira = html.escape(partes[0][len(prefixo_txt):])
        resultado.append(prefixo + f'<span class="c">{primeira}</span>')
        resultado += [f'<span class="c">{html.escape(p)}</span>' for p in partes[1:]]
    return resultado


FONTE = '15px/18px "Ubuntu Mono","DejaVu Sans Mono",monospace'
_LARGURA_CHAR: float | None = None


def largura_char(chrome: str) -> float:
    """Mede no próprio Chrome a largura de um caractere da fonte do terminal."""
    global _LARGURA_CHAR
    if _LARGURA_CHAR is None:
        teste = SAIDA / ".medida.html"
        teste.write_text(
            f'<!doctype html><meta charset="utf-8"><style>#p{{font:{FONTE};display:inline-block;'
            f'width:max-content;margin:0}}</style><pre id="p">{"M" * 200}</pre>'
            '<script>document.body.setAttribute("data-w",'
            'document.getElementById("p").getBoundingClientRect().width / 200)</script>',
            encoding="utf-8")
        saida = subprocess.run([chrome, "--headless=new", "--disable-gpu", f"--user-data-dir={SAIDA / '.chrome'}",
                                "--dump-dom", teste.as_uri()], capture_output=True, text=True, timeout=60).stdout
        teste.unlink(missing_ok=True)
        achado = re.search(r'data-w="([0-9.]+)"', saida)
        _LARGURA_CHAR = float(achado.group(1)) if achado else 8.3
    return _LARGURA_CHAR


def renderizar(sessao: Sessao, passos: list[tuple[str, str]], chrome: str) -> Path:
    corpo: list[str] = []
    for comando, saida in passos:
        corpo += linhas_comando(comando)
        if saida:
            corpo += ansi_para_html(saida)
    corpo.append(prompt_html() + '<span class="cursor"> </span>')
    larg_char, alt_linha = largura_char(chrome), 18
    largura = int(COLUNAS * larg_char + 2 * 14) + 2
    altura = int(len(corpo) * alt_linha + 2 * 12 + 34)
    titulo = html.escape(f"{USUARIO}@{HOST}: {DIR_TITULO}")
    pagina = f"""<!doctype html><meta charset="utf-8"><style>
html,body{{margin:0;background:{FUNDO};}}
.barra{{height:34px;background:#2d2d2d;color:#e6e6e6;font:600 13px "Ubuntu","Cantarell",sans-serif;
display:flex;align-items:center;justify-content:center;position:relative;border-bottom:1px solid #111}}
.botoes{{position:absolute;right:10px;top:7px;display:flex;gap:8px}}
.botoes span{{width:20px;height:20px;border-radius:50%;background:#444;color:#ddd;font:12px sans-serif;
display:flex;align-items:center;justify-content:center}}
.botoes span:last-child{{background:#e95420;color:#fff}}
pre{{margin:0;padding:12px 14px;font:{FONTE};color:{FRENTE};
white-space:pre;background:{FUNDO}}}
.u{{color:#8ae234;font-weight:bold}} .d{{color:#729fcf;font-weight:bold}} .c{{color:{FRENTE}}}
.cursor{{background:{FRENTE}}}
</style><div class="barra">{titulo}<div class="botoes"><span>–</span><span>□</span><span>✕</span></div></div>
<pre>{chr(10).join(corpo)}</pre>"""
    arquivo_html = SAIDA / f".{sessao.codigo}.html"
    arquivo_html.write_text(pagina, encoding="utf-8")
    destino = SAIDA / f"{sessao.codigo}_{sessao.nome}.png"
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                    f"--user-data-dir={SAIDA / '.chrome'}", "--force-device-scale-factor=2",
                    f"--window-size={largura},{altura}", f"--screenshot={destino}",
                    arquivo_html.as_uri()], capture_output=True, timeout=120)
    arquivo_html.unlink(missing_ok=True)
    return destino


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sessoes", nargs="*", help="códigos (T01…) ou nomes; padrão: todas")
    parser.add_argument("--listar", action="store_true")
    parser.add_argument("--so-png", action="store_true", help="re-renderiza a partir dos .json")
    args = parser.parse_args()
    if args.listar:
        for s in SESSOES:
            print(f"{s.codigo}  {s.nome:<20} {s.titulo}")
        return 0
    escolhidas = [s for s in SESSOES if not args.sessoes or s.codigo in args.sessoes or s.nome in args.sessoes]
    chrome = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")
    SAIDA.mkdir(parents=True, exist_ok=True)
    for sessao in escolhidas:
        dados = SAIDA / f"{sessao.codigo}_{sessao.nome}.json"
        if args.so_png:
            if not dados.is_file():
                continue
            passos = [tuple(p) for p in json.loads(dados.read_text(encoding="utf-8"))["passos"]]
        else:
            inicio = time.perf_counter()
            print(f">>> {sessao.codigo} {sessao.titulo} ...", flush=True)
            passos = executar(sessao)
            log = gravar_log(sessao, passos)
            print(log.read_text(encoding="utf-8"))
            print(f"    ({time.perf_counter() - inicio:.0f} s)", flush=True)
        if chrome:
            print(f"PNG: {renderizar(sessao, passos, chrome).relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
