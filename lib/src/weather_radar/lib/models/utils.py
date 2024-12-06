import re

CAMEL_TO_KEBAB = re.compile(r'(?<!^)(?=[A-Z])')
KEBAB_TO_CAMEL = re.compile(r'(?<!\A)-(?=[a-zA-Z])',re.X)


def to_camel_case(value):
    tokens = KEBAB_TO_CAMEL.split(value)
    response = tokens.pop(0).lower()
    for remain in tokens:
        response += remain.capitalize()
    return response


def to_kebab_case(value):
    return CAMEL_TO_KEBAB.sub('_', value).lower()


class BoundsError(Exception): ...
