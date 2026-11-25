"""Optional ONNX Runtime overlay over the Live app's shared CUDA runtime."""
import hashlib
import json
import sys

from .paths import ROOT
from .pear_runtime import base_identity

COMMAND = 'Run Setup_InstantHMR.cmd with RGB MoCap Live closed.'
LOCK = ROOT / 'installer/instanthmr-runtime.json'
RECEIPT = ROOT / '.setup/instanthmr-runtime.json'
RUNTIME = ROOT / 'runtime_instanthmr'
PACKAGE_FILES = (
    'flatbuffers/__init__.py', 'google/protobuf/__init__.py', 'onnx/__init__.py',
    'onnxruntime/__init__.py', 'onnxruntime/capi/onnxruntime_providers_cuda.dll',
    'onnxruntime/capi/onnxruntime_providers_shared.dll',
    'onnxruntime/capi/onnxruntime_pybind11_state.pyd',
)


def identity():
    return hashlib.sha256(LOCK.read_bytes()).hexdigest()


def configuration():
    data = json.loads(RECEIPT.read_text(encoding='utf8'))
    path = (RUNTIME / data['folder']).resolve()
    if (path.parent != RUNTIME.resolve() or not path.is_dir()
            or data['identity'] != identity() or data['base'] != base_identity()):
        raise ValueError('InstantHMR runtime needs setup.')
    for name in PACKAGE_FILES:
        stamp = data['files'][name]
        file = (path / name).resolve()
        if not file.is_relative_to(path):
            raise ValueError('Invalid InstantHMR package path.')
        stat = file.stat()
        if stamp != [stat.st_size, stat.st_mtime_ns]:
            raise ValueError('InstantHMR runtime changed.')
    return path


def status():
    try:
        configuration()
    except (OSError, ValueError, KeyError, TypeError):
        return False, COMMAND
    return True, 'InstantHMR CUDA runtime ready.'


def activate():
    try:
        path = configuration()
    except (OSError, ValueError, KeyError, TypeError):
        raise RuntimeError(COMMAND) from None
    sys.path.insert(0, str(path))
