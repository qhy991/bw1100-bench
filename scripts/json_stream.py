"""Read every complete top-level JSON value; never accept a partial prefix."""
import json


def read_json_stream(path):
    text = path.read_text()
    decoder = json.JSONDecoder()
    values = []
    offset = 0
    while offset < len(text):
        while offset < len(text) and text[offset] in ' \t\r\n':
            offset += 1
        if offset == len(text):
            break
        value, offset = decoder.raw_decode(text, offset)
        values.append(value)
    if not values:
        raise ValueError('Empty JSON stream')
    return values


GATEWAY_SCHEMA = 'bw1100-bench.hcu-admission.v1'


def author_diagnostic(text):
    """True only for unparseable text that cannot be, or hide, a gateway record.

    The gateway writes each record whole with json.dump into a new file, schema first, so a
    record caught mid-write is a prefix that starts with '{' and, once past its first key,
    names the schema. Text that starts with anything else and names neither the schema nor
    an admission_receipts declaration is an author's captured output (r4b: a probe's whole
    stdout redirected into timing.json, issue #42), not an unreleased admission.
    """
    stripped = text.lstrip(' \t\r\n')
    return (bool(stripped) and stripped[0] not in '{['
            and GATEWAY_SCHEMA not in text and 'admission_receipts' not in text)
