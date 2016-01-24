"""Utility functions."""
import numpy as np
import uuid


def expand_hex(hexval, width):
    """Zero-pad a hexadecimal representation out to a given number of places.

    Example:

    Parameters
    ----------
    hexval : str
        Hexadecimal representation, produced by hex().
    width : int
        Number of hexadecimal places to expand.

    Returns
    -------
    padded_hexval : str
        Zero-extended hexadecimal representation.

    Note: An error is raised if width is less than the number of hexadecimal
    digits required to represent the number.
    """
    chars = hexval[2:]
    if width < len(chars):
        raise ValueError(
            "Received: `hexval={}`. Width ({}) must be >= {}."
            "".format(hexval, width, len(chars)))
    y = list('0x' + '0' * int(width))
    y[-len(chars):] = list(chars)
    return "".join(y)


def index_to_hexkey(index, depth):
    """Convert an integer to a hex-key representation.

    Example: index_to_hexkey(843, 2) -> '03/4b'

    Parameters
    ----------
    index : int
        Integer index representation.
    depth : int
        Number of levels in the key (number of slashes plus one).

    Returns
    -------
    key : str
        Slash-separated hex-key.
    """
    hx = expand_hex(hex(int(index)), depth * 2)
    chars = zip(hx[::2], hx[1::2], '/' * int(len(hx) / 2))
    tmp = ''.join([''.join(d) for d in chars])
    return tmp[3:-1]


def uniform_hexgen(depth, width=256):
    """Generator to produce uniformly distributed hexkeys at a given depth.

    Deterministic and consistent, equivalent to a strided xrange() that yields
    strings like '04/1b/22' for depth=3, width=256.

    Parameters
    ----------
    depth : int
        Number of nodes in a single branch. See docstring in keyutil.py for
        more information.
    width : int
        Child nodes per parent. See docstring in keyutil.py for more
        information.

    Returns
    -------
    key : str
        Hexadecimal key path.
    """
    max_index = width ** depth
    index = 0
    for index in range(max_index):
        v = expand_hex(hex(index), depth * 2)
        hexval = "0x" + "".join([a + b for a, b in zip(v[-2:1:-2], v[:1:-2])])
        yield index_to_hexkey(int(hexval, 16), depth)
    raise ValueError("Unique keys exhausted.")


def unpack_entity_list(entities, filter_nulls=True):
    """Turn a list of entities into key-np.ndarray objects.

    TODO(ejhumphrey): Is this a bottleneck in the data
    bundling / presentation process?

    Parameters
    ----------
    entities: list of Entities
        Note that all Entities must have identical fields.

    Returns
    -------
    arrays: dict of np.ndarrays
        Values in 'arrays' are keyed by the fields of the Entities.
    """
    data = dict()
    for entity in entities:
        if entity is None and filter_nulls:
            continue
        if data == {}:
            data = dict([(k, list()) for k in entity.keys()])
        for k, v in entity.values().iteritems():
            data[k].append(v)

    for k in data:
        data[k] = np.asarray(data[k])
    return data


# def dump(obj, stash):
#     filtered_dict = dict()

#     for k, value in six.iteritems(obj.items()):
#         if k.startswith('_'):
#             continue

#         if isinstance(value, np.ndarray):
#             filtered_dict[k] = item.__json__
#         else:
#             filtered_dict[k] = item

#     return filtered_dict


def random_ndarray_generator(shape, loc=0, scale=1.0, max_items=None,
                             dtype=np.float64, seed=12345):
    """Produce a number of key-value, normally distributed ndarrays.

    Parameters
    ----------
    num_items : int
        Number of ndarrays to produce

    shape : array_like
        Shape of the ndarrays to produce.

    dtype : type, default=np.float64
        Datatype of the ndarrays.

    seed : int, default=12345
        Seed for the random number generator.

    Yields
    ------
    key, ndarray : str, np.ndarray
        Random value ndarray with a key.
    """
    rng = np.random.RandomState(seed)
    count = 0
    while True:
        if max_items is not None and count >= max_items:
            break
        yield uuid.uuid4(), rng.normal(loc, scale, shape).astype(dtype=dtype)
        count += 1
