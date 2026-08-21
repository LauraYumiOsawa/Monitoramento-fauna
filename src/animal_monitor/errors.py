class AnimalMonitorError(Exception):
    """Erro esperado e apresentável ao usuário."""


class ConfigurationError(AnimalMonitorError):
    """Configuração inválida ou incompleta."""


class DatasetValidationError(AnimalMonitorError):
    """Dataset incompatível com o formato esperado."""


class ModelOutputError(AnimalMonitorError):
    """Saída do modelo não reconhecida pelo pós-processamento."""
