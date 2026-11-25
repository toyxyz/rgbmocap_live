"""Pinned upstream InstantHMR graphs. The weights are installed separately."""
REPO = 'https://huggingface.co/momolesang/InstantHMR'
REVISION = 'fcde80ffe1181c946ef99251484c80083e8faede'

ASSETS = tuple(dict(key=key, name=name, relative='instanthmr/' + filename,
                    url=REPO, minimum_bytes=size, expected_sha256=digest,
                    download_url=f'{REPO}/resolve/{REVISION}/{filename}')
               for key, name, filename, size, digest in (
    ('instant_network', 'InstantHMR Medium Commercial', 'instanthmr-medium-commercial.onnx', 129345069,
     'f19d4492eba035337dc4fdc347cf3fe911570e73a765a70a35f9b32d7dec60dc'),
    ('instant_body', 'MHR Body Graph', 'mhr-body.onnx', 12676712,
     'eca2224d724f91b9e4f4bd7c86f6eafc74ab59ff0494bfb462e375cd04dac0fb'),
))
KEYS = ('instant_network', 'instant_body', 'pear_yolo')
