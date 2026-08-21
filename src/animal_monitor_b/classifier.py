"""Treinamento do classificador de espécie sobre recortes.

Por que classificar em vez de detectar as espécies direto:

* o dataset tem 207 caixas em 3 espécies e área relativa mediana de 0.47 — localizar o
  animal é fácil, distinguir a espécie é o que falha (o Modelo A confunde lobo-guará e
  tamanduá com anta);
* um classificador aproveita backbone pré-treinado em ImageNet com **transferência forte**
  e aumentos agressivos sobre recortes, o que é muito mais eficiente em dados do que
  treinar cabeças de detecção multiclasse do zero;
* a classe `desconhecido` dá ao sistema a opção de recusar, que a cabeça YOLO não tem.

Estratégia de treino em duas fases: sonda linear com backbone congelado (estabiliza a
cabeça sem destruir os pesos ImageNet) e depois ajuste fino apenas dos últimos blocos.
"""

from __future__ import annotations

import json
import platform
import random
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np

from animal_monitor.dataset import IMAGE_SUFFIXES
from animal_monitor.errors import ConfigurationError
from animal_monitor_b.augment import augment_crop, resize_square, to_tensor
from animal_monitor_b.config import ClassifierTrainConfig
from animal_monitor_b.metrics import classification_report

# MobileNetV3 tem o melhor F1 em FP32, mas hard-swish e squeeze-excite quantizam mal.
# MobileNetV2 e ShuffleNetV2 usam ReLU/ReLU6 e sobrevivem melhor ao INT8, que e o formato
# exigido pela NPU de placas de baixo custo. Os dois caminhos ficam disponiveis para o
# experimento comparativo.
SUPPORTED_BACKBONES = (
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "mobilenet_v2",
    "shufflenet_v2_x0_5",
    "shufflenet_v2_x1_0",
)


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch não está instalado. Instale o projeto com o extra 'modelo-b'."
        ) from exc
    return torch


def _torchvision():
    try:
        import torchvision
    except ImportError as exc:
        raise RuntimeError(
            "torchvision não está instalado. Instale o projeto com o extra 'modelo-b'."
        ) from exc
    return torchvision


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def build_model(backbone: str, class_count: int, pretrained: bool = True):
    """Cria a rede leve com a cabeça redimensionada para as classes do projeto."""
    torch = _torch()
    torchvision = _torchvision()
    if backbone not in SUPPORTED_BACKBONES:
        raise ConfigurationError(
            f"Backbone '{backbone}' não suportado. Use um de: {', '.join(SUPPORTED_BACKBONES)}."
        )
    factory = getattr(torchvision.models, backbone)
    model = factory(weights="DEFAULT" if pretrained else None)
    if backbone.startswith("mobilenet"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_features, class_count)
    else:
        model.fc = torch.nn.Linear(model.fc.in_features, class_count)
    return model


def _feature_modules(model, backbone: str) -> list:
    if backbone.startswith("mobilenet"):
        return list(model.features)
    return [model.conv1, model.stage2, model.stage3, model.stage4, model.conv5]


def set_trainable(model, backbone: str, unfreeze_from: int | None) -> int:
    """Congela o backbone e libera a cabeça.

    `unfreeze_from=None` mantém só a cabeça treinável (sonda linear); um inteiro libera
    também os blocos de índice maior ou igual a ele. Retorna o nº de parâmetros treináveis.
    """
    blocks = _feature_modules(model, backbone)
    for parameter in model.parameters():
        parameter.requires_grad = False
    head = model.classifier if hasattr(model, "classifier") else model.fc
    for parameter in head.parameters():
        parameter.requires_grad = True
    if unfreeze_from is not None:
        for block in blocks[min(unfreeze_from, len(blocks)) :]:
            for parameter in block.parameters():
                parameter.requires_grad = True
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def discover_labels(crops_dir: Path, split: str = "train") -> list[str]:
    split_dir = crops_dir / split
    if not split_dir.is_dir():
        raise ConfigurationError(f"Diretório de recortes não encontrado: {split_dir}")
    labels_file = crops_dir / "labels.json"
    if labels_file.is_file():
        labels = json.loads(labels_file.read_text(encoding="utf-8"))
        if isinstance(labels, list) and labels:
            return [str(label) for label in labels]
    return sorted(path.name for path in split_dir.iterdir() if path.is_dir())


class CropCache:
    """Carrega todos os recortes de um split na memória.

    São poucas centenas de imagens pequenas; manter em RAM elimina I/O por época e
    dispensa `num_workers` (que foi justamente o que causou OOM no Modelo A).
    """

    def __init__(self, crops_dir: Path, split: str, labels: list[str]) -> None:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("OpenCV não está instalado.") from exc
        self.labels = labels
        self.images: list[np.ndarray] = []
        self.targets: list[int] = []
        self.paths: list[Path] = []
        split_dir = crops_dir / split
        for index, label in enumerate(labels):
            label_dir = split_dir / label
            if not label_dir.is_dir():
                continue
            for path in sorted(label_dir.iterdir()):
                if path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                image = cv2.imread(str(path))
                if image is None:
                    continue
                self.images.append(image)
                self.targets.append(index)
                self.paths.append(path)
        if not self.images:
            raise ConfigurationError(f"Nenhum recorte encontrado em {split_dir}.")
        self.targets_array = np.asarray(self.targets, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.images)

    def counts(self) -> dict[str, int]:
        return {
            label: int((self.targets_array == index).sum())
            for index, label in enumerate(self.labels)
        }

    def batches(
        self,
        batch_size: int,
        input_size: int,
        rng: random.Random | None,
        shuffle: bool,
        sample_weights: np.ndarray | None = None,
    ):
        """Gera lotes (tensor NCHW, alvos). Com `sample_weights`, amostra com reposição."""
        count = len(self.images)
        generator = np.random.default_rng(rng.randrange(2**32) if rng is not None else 0)
        if sample_weights is not None:
            probabilities = sample_weights / sample_weights.sum()
            order = generator.choice(count, size=count, replace=True, p=probabilities)
        else:
            order = np.arange(count)
            if shuffle:
                generator.shuffle(order)
        for start in range(0, count, batch_size):
            indices = order[start : start + batch_size]
            tensors = []
            for index in indices:
                image = self.images[int(index)]
                if rng is not None:
                    processed = augment_crop(image, rng, input_size)
                else:
                    processed = resize_square(image, input_size)
                tensors.append(to_tensor(processed))
            yield np.stack(tensors, axis=0), self.targets_array[indices]


def _predict(model, cache: CropCache, input_size: int, batch_size: int) -> np.ndarray:
    torch = _torch()
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for tensors, _ in cache.batches(batch_size, input_size, None, shuffle=False):
            logits = model(torch.from_numpy(tensors))
            outputs.append(torch.softmax(logits, dim=1).numpy())
    return np.concatenate(outputs, axis=0)


def _run_phase(
    model,
    train_cache: CropCache,
    validation_cache: CropCache,
    config: ClassifierTrainConfig,
    epochs: int,
    learning_rate: float,
    unfreeze_from: int | None,
    rng: random.Random,
    sample_weights: np.ndarray | None,
    history: list[dict[str, float]],
    best: dict[str, object],
    phase_name: str,
) -> None:
    if epochs <= 0:
        return
    torch = _torch()
    trainable = set_trainable(model, config.backbone, unfreeze_from)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for tensors, targets in train_cache.batches(
            config.batch_size,
            config.input_size,
            rng,
            shuffle=True,
            sample_weights=sample_weights,
        ):
            optimizer.zero_grad(set_to_none=True)
            logits = model(torch.from_numpy(tensors))
            loss = criterion(logits, torch.from_numpy(targets))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(targets)
            seen += len(targets)
        scheduler.step()

        probabilities = _predict(model, validation_cache, config.input_size, config.batch_size)
        report = classification_report(
            validation_cache.targets_array, probabilities.argmax(axis=1), train_cache.labels
        )
        macro_f1 = float(report["macro_f1"])
        entry = {
            "phase": phase_name,
            "epoch": epoch,
            "train_loss": total_loss / max(1, seen),
            "val_macro_f1": macro_f1,
            "val_accuracy": float(report["accuracy"]),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "trainable_parameters": trainable,
        }
        history.append(entry)
        print(
            f"[{phase_name}] época {epoch:02d}/{epochs} "
            f"perda={entry['train_loss']:.4f} "
            f"val_macroF1={macro_f1:.4f} val_acc={entry['val_accuracy']:.4f}",
            flush=True,
        )

        if macro_f1 > float(best["macro_f1"]):
            best["macro_f1"] = macro_f1
            best["state"] = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best["report"] = report
            best["epoch"] = f"{phase_name}:{epoch}"
            best["stale"] = 0
        else:
            best["stale"] = int(best["stale"]) + 1
            if int(best["stale"]) >= config.patience:
                print(f"[{phase_name}] parada antecipada após {config.patience} épocas.")
                return


def train_classifier(config_path: str | Path) -> dict[str, object]:
    """Treina, avalia e salva o classificador de espécie."""
    torch = _torch()
    config = ClassifierTrainConfig.from_yaml(config_path)
    if config.threads > 0:
        torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    rng = random.Random(config.seed)
    np.random.seed(config.seed)

    labels = discover_labels(config.crops_dir)
    train_cache = CropCache(config.crops_dir, "train", labels)
    validation_cache = CropCache(config.crops_dir, "val", labels)

    sample_weights = None
    if config.balanced_sampling:
        counts = np.bincount(train_cache.targets_array, minlength=len(labels)).astype(np.float64)
        inverse = np.divide(1.0, counts, out=np.zeros_like(counts), where=counts > 0)
        sample_weights = inverse[train_cache.targets_array]

    model = build_model(config.backbone, len(labels), pretrained=True)
    total_parameters = int(sum(p.numel() for p in model.parameters()))
    history: list[dict[str, float]] = []
    best: dict[str, object] = {"macro_f1": -1.0, "state": None, "report": None, "stale": 0}

    started = datetime.now(timezone.utc)
    _run_phase(
        model,
        train_cache,
        validation_cache,
        config,
        config.epochs_head,
        config.learning_rate_head,
        None,
        rng,
        sample_weights,
        history,
        best,
        "sonda-linear",
    )
    best["stale"] = 0
    _run_phase(
        model,
        train_cache,
        validation_cache,
        config,
        config.epochs_finetune,
        config.learning_rate_finetune,
        config.unfreeze_from,
        rng,
        sample_weights,
        history,
        best,
        "ajuste-fino",
    )
    if best["state"] is None:  # pragma: no cover - só ocorre se nenhuma época rodar
        raise RuntimeError("O treinamento não produziu nenhum estado válido.")
    model.load_state_dict(best["state"])
    duration = (datetime.now(timezone.utc) - started).total_seconds()

    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.output_dir / "species_classifier.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "labels": labels,
            "backbone": config.backbone,
            "input_size": config.input_size,
            "normalization": "imagenet",
        },
        checkpoint_path,
    )

    result: dict[str, object] = {
        "checkpoint": str(checkpoint_path),
        "labels": labels,
        "backbone": config.backbone,
        "input_size": config.input_size,
        "total_parameters": total_parameters,
        "best_epoch": best["epoch"],
        "duration_s": duration,
        "train_counts": train_cache.counts(),
        "val_counts": validation_cache.counts(),
        "val_report": best["report"],
        "history": history,
        "config": {key: str(value) for key, value in asdict(config).items()},
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": _version("torch"),
            "torchvision": _version("torchvision"),
        },
    }

    test_dir = config.crops_dir / "test"
    if test_dir.is_dir():
        test_cache = CropCache(config.crops_dir, "test", labels)
        probabilities = _predict(model, test_cache, config.input_size, config.batch_size)
        result["test_counts"] = test_cache.counts()
        result["test_report"] = classification_report(
            test_cache.targets_array, probabilities.argmax(axis=1), labels
        )

    (config.output_dir / "classifier_training.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result
