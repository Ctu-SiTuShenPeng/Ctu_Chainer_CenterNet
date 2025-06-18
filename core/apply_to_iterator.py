import warnings
from core.unzip import unzip

def apply_to_iterator(func, iterator, n_input=1, hook=None, comm=None):

    if comm is None or comm.rank == 0:
        in_values, out_values, rest_values = unzip(
            _apply(func, iterator, n_input, hook, comm))

        # in_values: iter of ([in_val0], [in_val1], ...)
        #     -> (iter of in_val0, iter of in_val1, ...)
        in_values = tuple(map(_flatten, unzip(in_values)))

        # out_values: iter of ([out_val0], [out_val1], ...)
        #     -> (iter of out_val0, iter of out_val1, ...)
        out_values = tuple(map(_flatten, unzip(out_values)))

        # rest_values: iter of ([rest_val0], [rest_val1], ...)
        #     -> (iter of rest_val0, iter of rest_val1, ...)
        rest_values = tuple(map(_flatten, unzip(rest_values)))

        return in_values, out_values, rest_values
    else:
        # dummy loop to proceed generator
        for _ in _apply(func, None, n_input, None, comm):
            pass


def _apply(func, iterator, n_input, hook, comm):
    if comm is None:
        comm_size = 1
        comm_rank = 0
    else:
        comm_size = comm.size
        comm_rank = comm.rank

    batchsize_checked = False
    while True:
        if comm_rank == 0:
            try:
                batch = next(iterator)
                # batch: [(in_val0, in_val1, ... , rest_val0, rest_val1, ...)]
                #     or [in_val]

                q = len(batch) // comm_size
                r = len(batch) % comm_size

                if not batchsize_checked:
                    if not r == 0:
                        warnings.warn(
                            'The batchsize of the given iterator ({}) is not '
                            'a multiple of the number of workers ({}). '
                            'The total batchsize among all workers should be '
                            'specified and current setting will have a bad '
                            'effect on performace. '
                            .format(len(batch), comm_size),
                            RuntimeWarning)
                    batchsize_checked = True

                in_values = []
                rest_values = []
                in_values_locals = [[] for _ in range(comm_size)]
                for i, sample in enumerate(batch):
                    if i < (q + 1) * r:
                        k = i // (q + 1)
                    else:
                        k = (i - r) // q

                    if isinstance(sample, tuple):
                        in_values.append(sample[0:n_input])
                        rest_values.append(sample[n_input:])
                        in_values_locals[k].append(sample[0:n_input])
                    else:
                        in_values.append((sample,))
                        rest_values.append(())
                        in_values_locals[k].append((sample,))

            except StopIteration:
                in_values_locals = [None] * comm_size

        else:
            in_values_locals = None

        if comm is None:
            in_values_local = in_values_locals[0]
        else:
            in_values_local = comm.mpi_comm.scatter(in_values_locals)

        if in_values_local is None:
            break
        elif len(in_values_local) == 0:
            out_values_local = None
        else:
            # in_values_local: [(in_val0, in_val1, ...)]
            #     ->  ([in_val0], [in_val1], ...)
            in_values_local = tuple(list(v) for v in zip(*in_values_local))

            # out_values_local: ([out_val0], [out_val1], ...) or [out_val]
            out_values_local = func(*in_values_local)
            if not isinstance(out_values_local, tuple):
                # out_values_local: [out_val] -> ([out_val],)
                out_values_local = out_values_local,

        if comm is None:
            out_values_locals = [out_values_local]
        else:
            out_values_locals = comm.gather_obj(out_values_local)

        if comm_rank == 0:
            out_values = out_values_locals.pop(0)
            for out_values_local in out_values_locals:
                if out_values_local is None:
                    break
                for out_val, out_val_local in zip(
                        out_values, out_values_local):
                    out_val += out_val_local

            # in_values: [(in_val0, in_val1, ...)]
            #     ->  ([in_val0], [in_val1], ...)
            in_values = tuple(list(v) for v in zip(*in_values))

            # rest_values: [(rest_val0, rest_val1, ...)]
            #     -> ([rest_val0], [rest_val1], ...)
            rest_values = tuple(list(v) for v in zip(*rest_values))

            if hook:
                hook(in_values, out_values, rest_values)

            yield in_values, out_values, rest_values


def _flatten(iterator):
    return (sample for batch in iterator for sample in batch)
