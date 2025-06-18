import numbers,six,chainer
import numpy as np

def _is_iterable(x):
    if isinstance(x, str):
        return False
    return hasattr(x, '__iter__')

def _as_tuple(t):
    if _is_iterable(t):
        return tuple(t)
    else:
        return t,

def _bool_to_indices(indices, len_):
    true_indices = []
    for i, index in enumerate(indices):
        if isinstance(index, (bool, np.bool_)):
            if index:
                true_indices.append(i)
        else:
            return indices

    if not len(indices) == len_:
        raise ValueError('The number of booleans is different from the length of dataset')
    return true_indices

def _as_key_indices(keys, key_names):
    key_names = _as_tuple(key_names)
    keys = _bool_to_indices(_as_tuple(keys), len(key_names))

    for key in keys:
        if isinstance(key, numbers.Integral):
            key_index = key
            if key_index < 0:
                key_index += len(key_names)
            if key_index not in range(0, len(key_names)):
                raise IndexError('index {} is out of bounds for keys with size {}'.format(key, len(key_names)))
        else:
            try:
                key_index = key_names.index(key)
            except ValueError:
                raise KeyError('{} does not exists'.format(key))
        yield key_index

class SliceableDataset(chainer.dataset.DatasetMixin):
    def __len__(self):
        raise NotImplementedError

    @property
    def keys(self):
        raise NotImplementedError

    def get_example_by_keys(self, index, key_indices):
        raise NotImplementedError

    def get_example(self, index):
        if isinstance(self.keys, tuple):
            return self.get_example_by_keys(
                index, tuple(range(len(self.keys))))
        else:
            return self.get_example_by_keys(index, (0,))[0]

    @property
    def slice(self):
        return SliceHelper(self)

    def __iter__(self):
        return (self.get_example(i) for i in six.moves.range(len(self)))

class SlicedDataset(SliceableDataset):
    def __init__(self, dataset, indices, key_indices):
        self._dataset = dataset
        self._indices = indices
        self._key_indices = key_indices

    def __len__(self):
        if isinstance(self._indices, slice):
            start, end, step = self._indices.indices(len(self._dataset))
            return len(range(start, end, step))
        else:
            return len(self._indices)

    @property
    def keys(self):
        keys = _as_tuple(self._dataset.keys)
        if isinstance(self._key_indices, tuple):
            return tuple(keys[key_index] for key_index in self._key_indices)
        else:
            return keys[self._key_indices]

    def get_example_by_keys(self, index, key_indices):
        if isinstance(key_indices, tuple):
            key_indices = tuple(
                _as_tuple(self._key_indices)[key_index]
                for key_index in key_indices)
        else:
            key_indices = _as_tuple(self._key_indices)[key_indices]

        if isinstance(self._indices, slice):
            start, _, step = self._indices.indices(len(self._dataset))
            return self._dataset.get_example_by_keys(
                start + index * step, key_indices)
        else:
            return self._dataset.get_example_by_keys(
                self._indices[index], key_indices)

class SliceHelper(object):
    def __init__(self, dataset):
        self._dataset = dataset

    def __getitem__(self, args):
        if isinstance(args, tuple):
            indices, keys = args
        else:
            indices = args
            keys = self._dataset.keys

        if not isinstance(indices, slice):
            indices = _bool_to_indices(indices, len(self._dataset))
        key_indices = tuple(_as_key_indices(keys, self._dataset.keys))
        return_tuple = _is_iterable(keys)

        return SlicedDataset(
            self._dataset, indices,
            tuple(key_indices) if return_tuple else key_indices[0])

class SliceableDataset(chainer.dataset.DatasetMixin):
    def __len__(self):
        raise NotImplementedError

    @property
    def keys(self):
        raise NotImplementedError

    def get_example_by_keys(self, index, key_indices):
        raise NotImplementedError

    def get_example(self, index):
        if isinstance(self.keys, tuple):
            return self.get_example_by_keys(
                index, tuple(range(len(self.keys))))
        else:
            return self.get_example_by_keys(index, (0,))[0]

    @property
    def slice(self):
        return SliceHelper(self)

    def __iter__(self):
        return (self.get_example(i) for i in six.moves.range(len(self)))

class GetterDataset(SliceableDataset):
    def __init__(self):
        self._keys = []
        self._getters = []
        self._return_tuple = True

    def __len__(self):
        raise NotImplementedError

    @property
    def keys(self):
        if self._return_tuple:
            return tuple(key for key, _, _ in self._keys)
        else:
            return self._keys[0][0]

    @keys.setter
    def keys(self, keys):
        self._keys = [self._keys[key_index]
                      for key_index in _as_key_indices(keys, self.keys)]
        self._return_tuple = _is_iterable(keys)

    def add_getter(self, keys, getter):
        self._getters.append(getter)
        if _is_iterable(keys):
            for key_index, key in enumerate(keys):
                self._keys.append((key, len(self._getters) - 1, key_index))
        else:
            self._keys.append((keys, len(self._getters) - 1, None))

    def get_example_by_keys(self, index, key_indices):
        example = []
        cache = {}
        for key_index in key_indices:
            _, getter_index, key_index = self._keys[key_index]
            if getter_index not in cache:
                cache[getter_index] = self._getters[getter_index](index)
            if key_index is None:
                example.append(cache[getter_index])
            else:
                example.append(cache[getter_index][key_index])
        return tuple(example)
