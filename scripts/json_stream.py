"""Read every complete top-level JSON value; never accept a partial prefix."""
import json
import re


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
_JOB_ID = re.compile(r'"job_id"\s*:\s*"([^"]+)"')


def author_diagnostic(path):
    """Job ids an author's captured output names, or None when it may be a gateway record.

    The r4b case (issue #42): a probe's whole stdout, including the gateway's final
    {"job_id", "status", "exit_code"} line, redirected into a .json file. Such text is
    admissible only if it decodes as UTF-8, does not start like a JSON value (the gateway
    writes each record whole, schema first, so a torn record still starts with '{'), names
    neither the admission schema nor an admission_receipts declaration, and names at least
    one job. The caller must still show every named job released by its own terminal; text
    that names no job, or a job without one, withholds the release as before.
    """
    try:
        text = path.read_bytes().decode('utf-8')
    except (OSError, UnicodeDecodeError):
        return None
    stripped = text.lstrip(' \t\r\n')
    if (not stripped or stripped[0] in '{[' or GATEWAY_SCHEMA in text
            or 'admission_receipts' in text):
        return None
    jobs = set(_JOB_ID.findall(text))
    return jobs or None
