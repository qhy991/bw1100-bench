"""A wrong runtime mapping must fail before the candidate process starts."""
import importlib.util
from pathlib import Path
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location('hcu_identity', str(
    Path(__file__).resolve().parents[1] / 'scripts/hcu_device_identity.py'))
identity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(identity)


class Hip:
    def __init__(self, count=1, pci=b'0000:56:00.0'):
        self.count, self.pci = count, pci

    def hipGetDeviceCount(self, pointer):
        pointer._obj.value = self.count
        return 0

    def hipDeviceGetPCIBusId(self, buffer, length, device):
        buffer.value = self.pci
        return 0


class IdentityTests(unittest.TestCase):
    def check(self, hip):
        with mock.patch.object(identity.ctypes, 'CDLL', return_value=hip), \
             mock.patch.object(identity.sys, 'argv', ['identity', '0000:56:00.0']):
            return identity.main()

    def test_exact_assigned_device_is_accepted(self):
        self.assertEqual(self.check(Hip()), 0)

    def test_wrong_card_or_multiple_cards_are_refused(self):
        self.assertEqual(self.check(Hip(pci=b'0000:9f:00.0')), 78)
        self.assertEqual(self.check(Hip(count=7)), 78)


if __name__ == '__main__':
    unittest.main()
