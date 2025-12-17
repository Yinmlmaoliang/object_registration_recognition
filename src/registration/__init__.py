"""
Object Registration Module.

This module provides functionality for registering objects from handheld scenes
and building template libraries for later recognition.
"""

from .object_registrar import ObjectRegistrar, RegistrationResult

__all__ = [
    "ObjectRegistrar",
    "RegistrationResult",
]
