import collections

class BufferedIterator(object):
    def __init__(self, iterator, buffers, index):
        self.iterator = iterator
        self.buffers = buffers
        self.index = index

    def __del__(self):
        self.buffers[self.index] = None

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return self.buffers[self.index].popleft()
        except IndexError:
            values = next(self.iterator)
            for buf, val in zip(self.buffers, values):
                # skip a value if the correponding iterator is deleted.
                if buf is not None:
                    buf.append(val)
            return self.buffers[self.index].popleft()

    next = __next__

def unzip(iterable):
    iterator = iter(iterable)
    values = next(iterator)
    buffers = [collections.deque((val,)) for val in values]
    return tuple(
        BufferedIterator(iterator, buffers, index)
        for index in range(len(buffers)))
