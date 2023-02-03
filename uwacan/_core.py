import numpy as np
import collections.abc
import abc
import pendulum

def _sanitize_datetime_input(input):
    """Sanitize datetimes to the same internal format.

    This is not really an outwards-facing function. The main use-case is
    to make sure that we have `pendulum.DateTime` objects to work with
    internally.
    It's recommended that users use nice datetimes instead of strings,
    but sometimes a user will pass a string somewhere and then we'll try to
    parse it.
    """
    try:
        return pendulum.instance(input)
    except ValueError as err:
        if 'instance() only accepts datetime objects.' in str(err):
            pass
        else:
            raise
    try:
        return pendulum.from_timestamp(input)
    except TypeError as err:
        if 'object cannot be interpreted as an integer' in str(err):
            pass
        else:
            raise
    return pendulum.parse(input)


class TimePeriod:
    def __init__(self, start=None, stop=None, center=None, duration=None):
        if start is not None:
            start = _sanitize_datetime_input(start)
        if stop is not None:
            stop = _sanitize_datetime_input(stop)
        if center is not None:
            center = _sanitize_datetime_input(center)

        if None not in (start, stop):
            _start = start
            _stop = stop
            start = stop = None
        elif None not in (center, duration):
            _start = center - pendulum.duration(seconds=duration / 2)
            _stop = center + pendulum.duration(seconds=duration / 2)
            center = duration = None
        elif None not in (start, duration):
            _start = start
            _stop = start + pendulum.duration(seconds=duration)
            start = duration = None
        elif None not in (stop, duration):
            _stop = stop
            _start = stop - pendulum.duration(seconds=duration)
            stop = duration = None
        elif None not in (start, center):
            _start = start
            _stop = start + (center - start) / 2
            start = center = None
        elif None not in (stop, center):
            _stop = stop
            _start = stop - (stop - center) / 2
            stop = center = None
        else:
            raise TypeError('Needs two of the input arguments to determine time range.')

        if (start, stop, center, duration) != (None, None, None, None):
            raise TypeError('Cannot input more than two input arguments to a time window!')

        self._period = pendulum.period(_start, _stop)

    def subperiod(self, time=None, /, *, start=None, stop=None, center=None, duration=None):
        if time is None:
            # Period specified with keyword arguments, convert to period.
            if (start, stop, center, duration).count(None) == 3:
                # Only one argument which has to be start or stop, fill the other from self.
                if start is not None:
                    period = type(self)(start=start, stop=self.stop)
                elif stop is not None:
                    period = type(self)(start=self.start, stop=stop)
                else:
                    raise TypeError('Cannot create subperiod from arguments')
            else:
                # The same types explicit arguments as the normal constructor
                period = type(self)(start=start, stop=stop, center=center, duration=duration)
        elif isinstance(time, type(self)):
            period = time
        elif isinstance(time, pendulum.period):
            period = type(self)(start=time.start, stop=time.stop)
        else:
            # It's not a period, so it shold be a single datetime. Parse or convert, check valitidy.
            time = _sanitize_datetime_input(time)
            if time not in self:
                raise ValueError(f"Received time outside of contained period")
            return time

        if period not in self:
            raise ValueError("Requested subperiod is outside contained time period")
        return period

    def __repr__(self):
        return f'TimePeriod(start={self.start}, stop={self.stop})'

    @property
    def start(self):
        return self._period.start

    @property
    def stop(self):
        return self._period.end

    @property
    def center(self):
        return self.start + pendulum.duration(seconds=self._period.total_seconds() / 2)

    def __contains__(self, other):
        if isinstance(other, type(self)):
            other = other._period
        if isinstance(other, pendulum.Period):
            return other.start in self._period and other.end in self._period
        return other in self._period


class TimeLeafin(abc.ABC):
    @property
    @abc.abstractmethod
    def time_period(self):
        ...

    @abc.abstractmethod
    def time_subperiod(self, time=None, /, *, start=None, stop=None, center=None, duration=None):
        ...


class TimeBranchin:
    def time_subperiod(self, time=None, /, *, start=None, stop=None, center=None, duration=None, **kwargs):
        children = {
            name: child.time_subperiod(time, start=start, stop=stop, center=center, duration=duration, **kwargs)
            for name, child in self.items()
        }
        return self.clone(children)


class Metadata(collections.UserDict):
    def __init__(self, *args, node=None, **metadata):
        self.node = node
        super().__init__(*args, **metadata)

    @property
    def _parent_metadata(self):
        if self.node._parent:
            return self.node._parent.metadata
        return {}

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            pass
        try:
            return self._parent_metadata[key]
        except KeyError:
            pass
        raise KeyError(f"Metadata '{key}' cannot be found for '{self.node.__class__.__name__}' object")

    def __contains__(self, key):
        return (super().__contains__(key)) or (key in self._parent_metadata)


class Node:
    def __init__(self, metadata=None):
        if isinstance(metadata, collections.abc.Mapping):
            metadata = Metadata(node=self, **metadata)
        elif metadata is None:
            metadata = Metadata(node=self)
        else:
            raise TypeError("Cannot use object of type '{metadata.__class__.__name__}' as metadata")
        self.metadata = metadata
        self._parent = None

    @property
    def _root(self):
        try:
            return self._parent._root
        except AttributeError:
            return self

    @property
    def _parent(self):
        return self.__parent

    @_parent.setter
    def _parent(self, parent):
        self.__parent = parent

    def apply(self, function, *args, **kwargs):
        if not isinstance(function, NodeOperation):
            function = LeafDataFunction(function)
        return function(self, *args, **kwargs)

    def reduce(self, function, dim, *args, **kwargs):
        if not isinstance(function, NodeOperation):
            function = Reduction(function)
        return function(self, dim=dim, *args, **kwargs)


class Leaf(Node):
    dims = tuple()

    @property
    def _leaf_type(self):
        return type(self)

    def _traverse(self, leaves=True, branches=True, root=True, topdown=True, return_depth=False):
        if not leaves:
            return
        yield (return_depth - 1, self) if return_depth else self

    @property
    def data(self):
        return self._data


class Branch(Node, collections.abc.MutableMapping):
    def __init__(self, dim, children=None, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self._children = {}
        if children is not None:
            for name, child in children.items():
                self[name] = child

    def clone(self, children):
        return type(self)(dim=self.dim, children=children, metadata=self.metadata)

    def __len__(self):
        return len(self._children)

    def __iter__(self):
        return iter(self._children)

    def __getitem__(self, key):
        return self._children[key]

    def __setitem__(self, key, value):
        if key in self:
            raise KeyError(f"Cannot overwrite data '{key}' in dimension {self.dim}")
        value._parent = self
        self._children[key] = value

    def __delitem__(self, key):
        if key not in self:
            raise KeyError(f"Non-existent key '{key}' cannot be removed from data in dimension {self.dim}")
        self._children[key]._parent = None
        del self._children[key]

    @property
    def _first(self):
        return next(iter(self.values()))

    @property
    def _leaf_type(self):
        return self._first._leaf_type

    def _traverse(self, leaves=True, branches=True, root=True, topdown=True, return_depth=False):
        if return_depth is True:
            return_depth = 1
        if root and topdown:
            yield self if not return_depth else (return_depth - 1, self)
        for child in self.values():
            yield from child._traverse(leaves=leaves, branches=branches, root=branches, topdown=topdown, return_depth=return_depth and return_depth + 1)
        if root and not topdown:
            yield self if not return_depth else (return_depth - 1, self)


class NodeOperation:
    def __init__(self, function):
        self.function = function

    def __call__(self, node, *args, **kwargs):
        if isinstance(node, Branch):
            children = {name: self(child, *args, **kwargs) for name, child in node.items()}
            return node.clone(children)

    @staticmethod
    def wrap_output(output, input_leaf):
        if isinstance(output, Leaf):
            new_leaf = output
            new_leaf.metadata = Metadata(input_leaf.metadata | new_leaf.metadata, node=new_leaf)
        else:
            new_leaf = input_leaf.clone(output)
        return new_leaf


class LeafDataFunction(NodeOperation):
    def __call__(self, leaf, *args, **kwargs):
        if out := super().__call__(leaf, *args, **kwargs):
            return out

        out = self.function(leaf.data, *args, **kwargs)
        return self.wrap_output(out, leaf)


class LeafFunction(NodeOperation):
    def __call__(self, leaf, *args, **kwargs):
        if out := super().__call__(leaf, *args, **kwargs):
            return out

        out = self.function(leaf, *args, **kwargs)
        return self.wrap_output(out, leaf)


class Reduction(NodeOperation):
    def __call__(self, root, dim, metadata_merger='keep equal', *args, **kwargs):
        # TODO: add something that allows these reductions to be used on reducing axes on leaves as well?
        if isinstance(root, Leaf):
            # TODO: if we want to use the same reduction function for reducing over "layers" in the tree but also reducing over the axes of a data leaf, we will have problems with the axes somewhere.
            return root.reduce(self.function, dim=dim, *args, **kwargs)

        if dim != root.dim:
            return super().__call__(root, dim, *args, metadata_merger=metadata_merger, **kwargs)

        if metadata_merger == 'keep equal':
            def metadata_merger(name, labels, data):
                try:
                    is_equal = np.allclose(data, data[0])
                except TypeError:
                    is_equal = all(x == data[0] for x in data)
                if is_equal:
                    return data[0]
        elif metadata_merger == 'stack':
            def metadata_merger(name, labels, data):
                return data
        elif metadata_merger == 'collect':
            def metadata_merger(name, labels, data):
                return {label: x for (label, x) in zip(labels, data)}

        new = root._first.copy()
        for new_node, *old_nodes in zip(
                new._traverse(leaves=True, branches=True, root=True, topdown=False, return_depth=False),
                *(
                    child._traverse(leaves=True, branches=True, root=True, topdown=False, return_depth=False)
                    for child in root.values()
                )
        ):
            if isinstance(new_node, Leaf):
                stacked = [item.data for item in old_nodes]
                # Sometimes error handling in python is extremely annoying...
                try:
                    data = self.function(stacked, *args, **kwargs, axis=0)
                except TypeError as err:
                    err_msg = str(err)
                    if not (
                        "got an unexpected keyword argument 'axis'" in err_msg
                        or "takes no keyword arguments" in err_msg
                        or "got multiple values for keyword argument 'axis'" in err_msg
                    ):
                        raise
                    try:
                        data = self.function(stacked, *args, **kwargs)
                    except Exception as err:
                        raise err from None
                new_node._data = data
            metadata = {}
            for key in new_node.metadata:
                stacked = [item.metadata[key] for item in old_nodes]
                labels = list(root)
                this_meta = metadata_merger(key, labels, stacked)
                if this_meta is not None:
                    metadata[key] = this_meta
            new_node.metadata = Metadata(node=new_node, **metadata)
        new.metadata = root.metadata | new.metadata  # This merge will promote metadata which survived the normal pruning, prioritizing the promoted data
        new.metadata.node = new
        return new
