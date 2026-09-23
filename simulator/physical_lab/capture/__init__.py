"""Capture-side adapters for physical-lab devices and screen evidence.

Implementation modules are imported explicitly by callers so importing one
adapter does not eagerly initialize optional OpenCV, ADB, or workflow code.
"""
