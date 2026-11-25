"""Install and verify InstantHMR's optional GPU runtime without changing PEAR."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mocap.instant_runtime import LOCK, PACKAGE_FILES, RECEIPT, RUNTIME, base_identity, identity, status
from mocap.runtime_registry import activate
from mocap.storage import atomic_json
from tools.pinned_download import download, valid
from tools.session_lock import app_lock

PROBE = '''
import sys
sys.path.insert(0, sys.argv[1])
import torch, numpy as np, onnx, onnxruntime as ort
assert torch.cuda.is_available(), 'InstantHMR requires a compatible NVIDIA GPU.'
from onnx import helper, TensorProto as T
x = helper.make_tensor_value_info('x', T.FLOAT, [1, 8, 8])
y = helper.make_tensor_value_info('y', T.FLOAT, [1, 8, 8])
model = helper.make_model(helper.make_graph([helper.make_node('MatMul', ['x', 'x'], ['y'])],
                          'CUDA validation', [x], [y]), opset_imports=[helper.make_opsetid('', 17)], ir_version=9)
options = ort.SessionOptions()
options.add_session_config_entry('session.disable_cpu_ep_fallback','1')
session = ort.InferenceSession(model.SerializeToString(), options, providers=['CUDAExecutionProvider'])
assert session.get_providers()[0] == 'CUDAExecutionProvider', 'CUDA execution provider did not initialize.'
np.testing.assert_equal(session.run(None, {'x': np.ones((1, 8, 8), np.float32)})[0], 8.)
print('Verified ONNX Runtime with CUDA execution.')
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel-dir', type=Path, help='Use pinned wheel files from this directory.')
    args = parser.parse_args()
    activate(ROOT, 'shared')
    with app_lock(ROOT):
        if status()[0]:
            print('InstantHMR CUDA runtime is already ready.')
            return
        pins = json.loads(LOCK.read_text(encoding='utf8'))
        cache = ROOT / '.setup/instanthmr-downloads'
        wheels = []
        for name, spec in pins.items():
            if args.wheel_dir:
                wheel = args.wheel_dir.resolve() / name
                if not valid(wheel, spec):
                    raise RuntimeError('Missing or mismatched wheel: ' + name)
            else:
                wheel = download(cache, name, spec, filename=name, allowed_hosts={'files.pythonhosted.org'})
            wheels.append(str(wheel))
        path = RUNTIME / ('packages-' + uuid4().hex)
        subprocess.run([sys.executable, '-I', '-B', '-m', 'pip', '--isolated',
                        '--disable-pip-version-check', 'install', '--no-index', '--no-deps',
                        '--target', str(path), *wheels], check=True)
        subprocess.run([sys.executable, '-I', '-B', '-c', PROBE, str(path)], check=True)
        files = {}
        for name in PACKAGE_FILES:
            stat = (path / name).stat()
            files[name] = [stat.st_size, stat.st_mtime_ns]
        atomic_json(RECEIPT, dict(folder=path.name, base=base_identity(), identity=identity(), files=files))
        print('InstantHMR runtime ready. Prepare the model files in Model Manager.')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from None
