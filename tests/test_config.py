import tempfile
import unittest
from pathlib import Path

from animal_monitor.config import RuntimeConfig, load_class_names
from animal_monitor.errors import ConfigurationError


class ConfigTest(unittest.TestCase):
    def test_loads_list_and_mapping_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            list_yaml = root / "list.yaml"
            list_yaml.write_text("names: [anta, onca]\n", encoding="utf-8")
            self.assertEqual(load_class_names(list_yaml), {0: "anta", 1: "onca"})
            map_yaml = root / "map.yaml"
            map_yaml.write_text("names:\n  0: anta\n  1: onca\n", encoding="utf-8")
            self.assertEqual(load_class_names(map_yaml), {0: "anta", 1: "onca"})

    def test_rejects_non_contiguous_classes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.yaml"
            path.write_text("names:\n  0: anta\n  2: onca\n", encoding="utf-8")
            with self.assertRaises(ConfigurationError):
                load_class_names(path)

    def test_runtime_resolves_output_paths_relative_to_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.yaml"
            path.write_text("output_dir: result\ndatabase: result/events.db\n", encoding="utf-8")
            config = RuntimeConfig.from_yaml(path)
            self.assertEqual(config.output_dir, (Path(directory) / "result").resolve())


if __name__ == "__main__":
    unittest.main()
