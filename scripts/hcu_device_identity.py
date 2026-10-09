"""Check assigned physical PCI before the candidate process can allocate tensors."""
import ctypes
import json
import sys


def main():
    try:
        expected = sys.argv[1].lower()
        hip = ctypes.CDLL('libamdhip64.so')
        count = ctypes.c_int()
        if hip.hipGetDeviceCount(ctypes.byref(count)) != 0 or count.value != 1:
            raise RuntimeError('exactly one runtime-assigned HIP device is required')
        pci = ctypes.create_string_buffer(64)
        if hip.hipDeviceGetPCIBusId(pci, len(pci), 0) != 0:
            raise RuntimeError('assigned device PCI query failed')
        observed = pci.value.decode().lower()
        if observed != expected:
            raise RuntimeError('expected PCI %s, observed %s' % (expected, observed))
        print(json.dumps({'runtime_device_identity': 'passed', 'pci_bus_id': observed,
                          'visible_devices': count.value}), flush=True)
        return 0
    except Exception as error:
        print('HCU_DEVICE_IDENTITY_FAILED: ' + str(error), flush=True)
        return 78


if __name__ == '__main__':
    sys.exit(main())
