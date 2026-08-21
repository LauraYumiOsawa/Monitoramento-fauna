from __future__ import annotations

import json
import os
import random
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from animal_monitor.dataset import IMAGE_SUFFIXES


@dataclass(frozen=True, slots=True)
class RawSample:
    image: Path
    label: Path | None
    relative_image: Path
    group: str


def _group_name(path: Path, pattern: re.Pattern[str] | None) -> str:
    if pattern is None:
        return path.stem
    match = pattern.search(path.stem)
    if not match:
        raise ValueError(f"O arquivo não corresponde ao group-regex: {path.name}")
    return match.group(1) if match.groups() else match.group(0)


def _copy_or_link(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(
            f"Destino já existe; divisão interrompida sem sobrescrever: {destination}"
        )
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "hardlink":
        os.link(source, destination)
    else:
        raise ValueError("mode deve ser 'copy' ou 'hardlink'.")


def split_dataset(
    images_dir: str | Path,
    labels_dir: str | Path,
    output_dir: str | Path,
    ratios: tuple[float, float, float] = (0.70, 0.20, 0.10),
    seed: int = 42,
    group_regex: str | None = None,
    mode: str = "copy",
) -> dict[str, object]:
    if len(ratios) != 3 or any(value <= 0 for value in ratios):
        raise ValueError("Informe três proporções positivas para train, val e test.")
    ratio_sum = sum(ratios)
    ratios = tuple(value / ratio_sum for value in ratios)
    images_root = Path(images_dir).expanduser().resolve()
    labels_root = Path(labels_dir).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    if not images_root.is_dir():
        raise FileNotFoundError(f"Diretório de imagens não encontrado: {images_root}")
    if not labels_root.is_dir():
        raise FileNotFoundError(f"Diretório de rótulos não encontrado: {labels_root}")
    pattern = re.compile(group_regex) if group_regex else None

    groups: dict[str, list[RawSample]] = defaultdict(list)
    for image in sorted(images_root.rglob("*")):
        if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        relative = image.relative_to(images_root)
        label = (labels_root / relative).with_suffix(".txt")
        sample = RawSample(
            image=image,
            label=label if label.is_file() else None,
            relative_image=relative,
            group=_group_name(relative, pattern),
        )
        groups[sample.group].append(sample)
    if not groups:
        raise ValueError(f"Nenhuma imagem encontrada em: {images_root}")
    if len(groups) < 3:
        raise ValueError("São necessários pelo menos três grupos para criar train, val e test.")

    shuffled_groups = list(groups.items())
    random.Random(seed).shuffle(shuffled_groups)
    shuffled_groups.sort(key=lambda item: len(item[1]), reverse=True)
    split_names = ("train", "val", "test")
    total_images = sum(len(samples) for _, samples in shuffled_groups)
    targets = {name: total_images * ratio for name, ratio in zip(split_names, ratios, strict=True)}
    assignments: dict[str, list[RawSample]] = {name: [] for name in split_names}
    group_assignments: dict[str, str] = {}

    # Preenche primeiro os três splits; depois aproxima cada meta sem quebrar grupos.
    for index, (group_name, samples) in enumerate(shuffled_groups):
        if index < len(split_names):
            chosen = split_names[index]
        else:
            chosen = max(
                split_names,
                key=lambda name: targets[name] - len(assignments[name]),
            )
        assignments[chosen].extend(samples)
        group_assignments[group_name] = chosen

    missing_labels = 0
    for split, samples in assignments.items():
        for sample in samples:
            image_destination = output_root / "images" / split / sample.relative_image
            _copy_or_link(sample.image, image_destination, mode)
            if sample.label is not None:
                label_destination = (
                    output_root / "labels" / split / sample.relative_image
                ).with_suffix(".txt")
                _copy_or_link(sample.label, label_destination, mode)
            else:
                missing_labels += 1

    manifest: dict[str, object] = {
        "seed": seed,
        "ratios": dict(zip(split_names, ratios, strict=True)),
        "mode": mode,
        "group_regex": group_regex,
        "source_images": str(images_root),
        "source_labels": str(labels_root),
        "output": str(output_root),
        "images": {name: len(samples) for name, samples in assignments.items()},
        "groups": group_assignments,
        "missing_labels": missing_labels,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest
