"""Lightweight stand-in for the legacy TF ``m3gnet`` package, importable by joblib worker processes.

The TF-based ``m3gnet`` package is not installable alongside current TensorFlow. Tests put the parent
directory on ``sys.path`` so both the test process and spawned workers import this fake.
"""
