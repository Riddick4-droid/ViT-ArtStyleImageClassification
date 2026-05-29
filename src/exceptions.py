
class ProjectException(Exception):
    """Base exception for all project‑specific errors."""
    pass

class DataIngestionError(ProjectException):
    """Raised when data download or validation fails."""
    pass

class DatasetError(ProjectException):
    """Raised for problems during dataset loading or processing."""
    pass

class ModelBuildError(ProjectException):
    """Raised when the model architecture configuration is invalid."""
    pass

class TrainingError(ProjectException):
    """Raised for unrecoverable training failures."""
    pass

class EvaluationError(ProjectException):
    """Raised when evaluation fails."""
    pass

class InferenceError(ProjectException):
    """Raised for errors during single‑image inference."""
    pass

class ConfigurationError(ProjectException):
    """Raised for missing or invalid config values."""
    pass