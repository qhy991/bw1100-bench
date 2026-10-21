"""Print the container image a Run is bound to, read from its own binding.

An author never supplies the image. R6 retyped the digest with one wrong character
and R9 pasted the usage placeholder `IMAGE`; both reached the gateway's fail-closed
`docker image inspect`, which wrote the HCU-wide stop file and drained every other
Run on that HCU. The value is already an exact field of the Run's binding, so the
profiling entry reads it here instead of taking it as an argument.
"""
import json
import re
import sys
from pathlib import Path

BINDINGS = ('campaign/development-binding.json', 'campaign/binding.json')
IMAGE = re.compile('sha256:[0-9a-f]{64}')


def pinned_image(root):
    for name in BINDINGS:
        path = Path(root) / name
        if path.is_file():
            image = json.loads(path.read_text()).get('image')
            if not isinstance(image, str) or IMAGE.fullmatch(image) is None:
                raise ValueError(name + ' does not bind an exact sha256 image')
            return image
    raise ValueError('no Run binding declares the pinned image: ' + ' or '.join(BINDINGS))


if __name__ == '__main__':
    try:
        print(pinned_image(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]))
    except ValueError as error:
        raise SystemExit(str(error))
