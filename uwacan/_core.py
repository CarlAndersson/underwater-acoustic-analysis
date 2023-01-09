import numpy as np
import collections.abc
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


class TimeWindow:
    @classmethod
    def from_slice(cls, slice):
        return cls(start=slice.start, stop=slice.stop)

    def __init__(self, start=None, stop=None, center=None, duration=None):
        if start is not None:
            start = _sanitize_datetime_input(start)
        if stop is not None:
            stop = _sanitize_datetime_input(stop)
        if center is not None:
            center = _sanitize_datetime_input(center)

        if None not in (start, stop):
            self._start = start
            self._stop = stop
            start = stop = None
        elif None not in (center, duration):
            self._start = center - pendulum.duration(seconds=duration / 2)
            self._stop = center + pendulum.duration(seconds=duration / 2)
            center = duration = None
        elif None not in (start, duration):
            self._start = start
            self._stop = start + pendulum.duration(seconds=duration)
            start = duration = None
        elif None not in (stop, duration):
            self._stop = stop
            self._start = stop - pendulum.duration(seconds=duration)
            stop = duration = None
        elif None not in (start, center):
            self._start = start
            self._stop = start + (center - start) / 2
            start = center = None
        elif None not in (stop, center):
            self._stop = stop
            self._start = stop - (stop - center) / 2
            stop = center = None

        if (start, stop, center, duration) != (None, None, None, None):
            raise ValueError('Cannot input more than two input arguments to a time window!')

    def __repr__(self):
        return f'TimeWindow(start={self.start}, stop={self.stop})'

    @property
    def start(self):
        return self._start

    @property
    def stop(self):
        return self._stop

    @property
    def duration(self):
        return (self.stop - self.start).total_seconds()

    @property
    def center(self):
        return self.start + pendulum.duration(seconds=self.duration / 2)

    def __contains__(self, other):
        if isinstance(other, TimeWindow):
            return (other.start >= self.start) and (other.stop <= self.stop)
        if hasattr(other, 'timestamp') and isinstance(other.timestamp, TimeWindow):
            return self.start <= other.timestamp <= self.stop
        try:
            other = _sanitize_datetime_input(other)
        except TypeError:
            raise TypeError(f"Cannot check if '{other.__class__.__name__}' object is within a time window")
        return self.start <= other <= self.stop


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

    def copy(self, new_node=None):
        new_meta = super().copy()
        new_meta.node = new_node if new_node is not None else self.node
        return new_meta


class Node:
    def __init__(self, metadata=None):
        if isinstance(metadata, Metadata):
            metadata = metadata.copy(new_node=self)
        elif metadata is None:
            metadata = Metadata(node=self)
        else:
            metadata = Metadata(node=self, **metadata)
        self.metadata = metadata
        self._parent = None

    @property
    def _root(self):
        try:
            return self._parent._root
        except AttributeError:
            return self

    def copy(self, _new_class=None):
        if _new_class is None:
            _new_class = type(self)
        obj = _new_class.__new__(_new_class)
        obj.metadata = self.metadata.copy(new_node=obj)
        obj._parent = None
        return obj

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

    def copy(self, _new_children=None, **kwargs):
        new = super().copy(**kwargs)
        if _new_children is None:
            _new_children = {name: child.copy() for name, child in self.items()}
        new.dim = self.dim
        new._children = _new_children
        for child in new.values():
            child._parent = new
        return new

    def __len__(self):
        return len(self._children)

    def __iter__(self):
        return iter(self._children)

    def __getitem__(self, key):
        if isinstance(key, slice):
            key = TimeWindow.from_slice(key)
        if isinstance(key, TimeWindow):
            children = {name: child[key] for name, child in self.items()}
            return self.copy(_new_children=children)
        return self._children[key]

    def __setitem__(self, key, value):
        if key in self:
            raise KeyError(f"Cannot overwrite data '{key}' in dimension {self.dim}")
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
            return node.copy(_new_children=children)


class LeafDataFunction(NodeOperation):
    def __call__(self, leaf, *args, **kwargs):
        if out := super().__call__(leaf, *args, **kwargs):
            return out

        out = self.function(leaf.data, *args, **kwargs)
        if isinstance(out, Leaf):
            new_leaf = out
            new_leaf.metadata = leaf.metadata | new_leaf.metadata  # Update in place not good since it will prioritize the old
            new_leaf.metadata.node = new_leaf
        else:
            new_leaf = leaf.copy()
            new_leaf._data = out

        return new_leaf


class LeafFunction(NodeOperation):
    def __call__(self, leaf, *args, **kwargs):
        if out := super().__call__(leaf, *args, **kwargs):
            return out

        new_leaf = self.function(leaf, *args, **kwargs)
        new_leaf.metadata = leaf.metadata | new_leaf.metadata
        new_leaf.metadata.node = new_leaf
        return new_leaf


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
