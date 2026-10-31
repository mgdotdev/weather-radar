import re

CAMEL_TO_KEBAB = re.compile(r'(?<!^)(?=[A-Z])')
KEBAB_TO_CAMEL = re.compile(r'(?<!\A)-(?=[a-zA-Z])',re.X)


def to_camel_case(value):
    tokens = KEBAB_TO_CAMEL.split(value)
    first = tokens.pop(0)
    # Lowercase only the first character so already-camelCase input (with no
    # hyphens to split on) is returned unchanged, e.g. "snowfallAmount" ->
    # "snowfallAmount" rather than "snowfallamount".
    response = first[:1].lower() + first[1:]
    for remain in tokens:
        # Uppercase only the leading character; preserve the rest so existing
        # casing in a token isn't flattened.
        response += remain[:1].upper() + remain[1:]
    return response


def to_kebab_case(value):
    return CAMEL_TO_KEBAB.sub('_', value).lower()


class BoundsError(Exception): ...
