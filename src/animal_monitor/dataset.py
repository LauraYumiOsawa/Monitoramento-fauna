from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from animal_monitor.config import load_class_names, load_yaml, resolve_from
from animal_monitor.errors import ConfigurationError

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True, slots=True)
class DatasetIssue:
    severity: str
    code: str
    message: str
    path: str | None = None


@dataclass(slots=True)
class SplitStatistics:
    images: int = 0
    missing_labels: int = 0
    empty_labels: int = 0
    boxes: int = 0
    class_boxes: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class DatasetReport:
    dataset_yaml: str
    dataset_root: str
    class_names: dict[int, str]
    splits: dict[str, SplitStatistics]
    issues: list[DatasetIssue]

    @property
    def ok(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "dataset_yaml": self.dataset_yaml,
            "dataset_root": self.dataset_root,
            "class_names": self.class_names,
            "splits": {name: asdict(stats) for name, stats in self.splits.items()},
            "issues": [asdict(issue) for issue in self.issues],
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False, indent=2)


def _expand_source(root: Path, source: str | Path) -> list[Path]:
    candidate = Path(source).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    if candidate.is_dir():
        return sorted(
            path
            for path in candidate.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    if candidate.is_file() and candidate.suffix.lower() == ".txt":
        images: list[Path] = []
        for raw_line in candidate.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            path = Path(line).expanduser()
            if not path.is_absolute():
                path = (candidate.parent / path).resolve()
            images.append(path)
        return sorted(images)
    if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
        return [candidate]
    return []


def _images_for_split(root: Path, value: object) -> list[Path]:
    if isinstance(value, (str, Path)):
        return _expand_source(root, value)
    if isinstance(value, list):
        images: list[Path] = []
        for item in value:
            if not isinstance(item, (str, Path)):
                raise ConfigurationError("Cada origem de split deve ser um caminho.")
            images.extend(_expand_source(root, item))
        return sorted(set(images))
    raise ConfigurationError("Cada split deve ser um caminho ou uma lista de caminhos.")


def label_path_for(image_path: Path, dataset_root: Path) -> Path:
    try:
        relative_parts = list(image_path.resolve().relative_to(dataset_root.resolve()).parts)
    except ValueError:
        relative_parts = list(image_path.parts)
    try:
        image_index = relative_parts.index("images")
        relative_parts[image_index] = "labels"
        label = Path(*relative_parts).with_suffix(".txt")
        return dataset_root / label if not label.is_absolute() else label
    except ValueError:
        return image_path.with_suffix(".txt")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_label(
    label_path: Path,
    class_names: dict[int, str],
) -> tuple[Counter[int], list[DatasetIssue]]:
    counts: Counter[int] = Counter()
    issues: list[DatasetIssue] = []
    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        tokens = line.split()
        if len(tokens) != 5:
            issues.append(
                DatasetIssue(
                    "error",
                    "invalid_label_columns",
                    f"Linha {line_number}: esperado 'classe x y largura altura', "
                    f"recebido {len(tokens)} valores.",
                    str(label_path),
                )
            )
            continue
        try:
            class_value = float(tokens[0])
            coordinates = [float(value) for value in tokens[1:]]
        except ValueError:
            issues.append(
                DatasetIssue(
                    "error",
                    "non_numeric_label",
                    f"Linha {line_number}: o rótulo contém valor não numérico.",
                    str(label_path),
                )
            )
            continue
        class_id = int(class_value)
        if class_value != class_id or class_id not in class_names:
            issues.append(
                DatasetIssue(
                    "error",
                    "invalid_class_id",
                    f"Linha {line_number}: ID de classe inválido: {tokens[0]}.",
                    str(label_path),
                )
            )
            continue
        x_center, y_center, width, height = coordinates
        if not all(0.0 <= value <= 1.0 for value in coordinates):
            issues.append(
                DatasetIssue(
                    "error",
                    "coordinates_out_of_range",
                    f"Linha {line_number}: coordenadas YOLO devem estar entre 0 e 1.",
                    str(label_path),
                )
            )
            continue
        if width <= 0.0 or height <= 0.0:
            issues.append(
                DatasetIssue(
                    "error",
                    "invalid_box_size",
                    f"Linha {line_number}: largura e altura devem ser maiores que zero.",
                    str(label_path),
                )
            )
            continue
        if x_center - width / 2 < 0 or x_center + width / 2 > 1:
            issues.append(
                DatasetIssue(
                    "warning",
                    "box_crosses_horizontal_boundary",
                    f"Linha {line_number}: a caixa ultrapassa a largura da imagem.",
                    str(label_path),
                )
            )
        if y_center - height / 2 < 0 or y_center + height / 2 > 1:
            issues.append(
                DatasetIssue(
                    "warning",
                    "box_crosses_vertical_boundary",
                    f"Linha {line_number}: a caixa ultrapassa a altura da imagem.",
                    str(label_path),
                )
            )
        counts[class_id] += 1
    return counts, issues


def validate_dataset(dataset_yaml: str | Path, minimum_boxes_per_class: int = 25) -> DatasetReport:
    yaml_path = Path(dataset_yaml).expanduser().resolve()
    content = load_yaml(yaml_path)
    class_names = load_class_names(yaml_path)
    root_value = content.get("path", ".")
    dataset_root = resolve_from(yaml_path, root_value)
    issues: list[DatasetIssue] = []
    split_stats: dict[str, SplitStatistics] = {}
    hashes: dict[str, list[tuple[str, Path]]] = defaultdict(list)

    for split in ("train", "val", "test"):
        source = content.get(split)
        if source is None:
            if split in {"train", "val"}:
                issues.append(
                    DatasetIssue(
                        "error", "missing_split", f"Split obrigatório ausente: {split}."
                    )
                )
            continue
        images = _images_for_split(dataset_root, source)
        stats = SplitStatistics(images=len(images))
        class_counts: Counter[int] = Counter()
        if not images:
            severity = "error" if split in {"train", "val"} else "warning"
            issues.append(
                DatasetIssue(
                    severity,
                    "empty_split",
                    f"Nenhuma imagem foi encontrada no split '{split}'.",
                )
            )

        for image_path in images:
            if not image_path.is_file():
                issues.append(
                    DatasetIssue(
                        "error", "missing_image", "Imagem listada não existe.", str(image_path)
                    )
                )
                continue
            hashes[_sha256(image_path)].append((split, image_path))
            label_path = label_path_for(image_path, dataset_root)
            if not label_path.is_file():
                stats.missing_labels += 1
                issues.append(
                    DatasetIssue(
                        "warning",
                        "missing_label",
                        "Sem rótulo; a imagem será tratada como exemplo negativo.",
                        str(image_path),
                    )
                )
                continue
            if not label_path.read_text(encoding="utf-8").strip():
                stats.empty_labels += 1
                continue
            counts, label_issues = _parse_label(label_path, class_names)
            class_counts.update(counts)
            issues.extend(label_issues)

        stats.boxes = sum(class_counts.values())
        stats.class_boxes = {class_names[index]: class_counts[index] for index in class_names}
        split_stats[split] = stats

        for class_id, class_name in class_names.items():
            count = class_counts[class_id]
            if split == "train" and count == 0:
                issues.append(
                    DatasetIssue(
                        "error",
                        "class_missing_from_train",
                        f"Classe sem caixas em treino: {class_name}.",
                    )
                )
            elif split in {"val", "test"} and count == 0 and images:
                issues.append(
                    DatasetIssue(
                        "warning",
                        "class_missing_from_evaluation",
                        f"Classe sem caixas em {split}: {class_name}.",
                    )
                )
            elif split == "train" and 0 < count < minimum_boxes_per_class:
                issues.append(
                    DatasetIssue(
                        "warning",
                        "few_training_boxes",
                        f"Classe '{class_name}' possui só {count} caixas em treino; "
                        f"mínimo de triagem: {minimum_boxes_per_class}.",
                    )
                )

        nonzero_counts = [count for count in class_counts.values() if count > 0]
        if split == "train" and len(nonzero_counts) > 1:
            imbalance = max(nonzero_counts) / min(nonzero_counts)
            if imbalance > 3.0:
                issues.append(
                    DatasetIssue(
                        "warning",
                        "class_imbalance",
                        f"Desbalanceamento de caixas em treino: razão {imbalance:.2f}:1.",
                    )
                )

    for duplicate_group in hashes.values():
        if len(duplicate_group) < 2:
            continue
        splits = {split for split, _ in duplicate_group}
        paths = ", ".join(str(path) for _, path in duplicate_group)
        if len(splits) > 1:
            issues.append(
                DatasetIssue(
                    "error",
                    "cross_split_duplicate",
                    "A mesma imagem aparece em splits diferentes "
                    f"({', '.join(sorted(splits))}): {paths}",
                )
            )
        else:
            issues.append(
                DatasetIssue("warning", "duplicate_image", f"Imagem repetida no split: {paths}")
            )

    return DatasetReport(
        dataset_yaml=str(yaml_path),
        dataset_root=str(dataset_root),
        class_names=class_names,
        splits=split_stats,
        issues=issues,
    )


def summarize_issue_counts(issues: Iterable[DatasetIssue]) -> dict[str, int]:
    counts = Counter(issue.severity for issue in issues)
    return {"errors": counts["error"], "warnings": counts["warning"]}
