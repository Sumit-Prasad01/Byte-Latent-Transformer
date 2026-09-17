"""Custom exception hierarchy for the Byte Latent Transformer (BLT) project.
"""

import sys
from typing import Optional


def error_message_detail(error: Exception, error_detail: sys) -> str:
    """Extract detailed error message with script name and line number."""
    _, _, exc_tb = error_detail.exc_info()
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        return f"Error occurred in script: [{file_name}] at line number: [{line_number}] error message: [{str(error)}]"
    return str(error)


class BLTException(Exception):
    """Base exception for all BLT project errors."""

    def __init__(self, error_message: str, error_detail: Optional[sys] = None):
        super().__init__(error_message)
        if error_detail is not None:
            self.error_message = error_message_detail(self, error_detail=error_detail)
        else:
            self.error_message = str(error_message)

    def __str__(self) -> str:
        return self.error_message


class NativeEngineError(BLTException):
    """Raised when native C++ kernel operations (.dll / .so) fail or produce invalid outputs."""
    pass


class DataPipelineError(BLTException):
    """Raised when data reading, preprocessing, sharding, or packing encounters an error."""
    pass


class ModelArchitectureError(BLTException):
    """Raised when shape mismatches, attention mask errors, or architectural violations occur."""
    pass


class CalibrationError(BLTException):
    """Raised when entropy threshold calibration fails to converge or receives invalid inputs."""
    pass


class TrainingConfigError(BLTException):
    """Raised when invalid configurations, GPU memory constraints, or hyperparameters are detected."""
    pass


class GenerationError(BLTException):
    """Raised when autoregressive sampling, streaming patching, or UTF-8 decoding fails."""
    pass
