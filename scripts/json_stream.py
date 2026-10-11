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
