import numpy as np
import collections.abc


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
        # if parent is None:
        #     self._parent_metadata = []
        # else:
        #     self._parent_metadata = list(parent._metadata.keys()) + parent._parent_metadata

        # for child in self._children:
        #     child._parent = self  # Updates the list of metadata in the children

    def apply(self, function, *args, **kwargs):
        if not isinstance(function, NodeOperation):
            function = LeafDataFunction(function)
        return function(self, *args, **kwargs)

    def reduce(self, function, axis, *args, **kwargs):
        if not isinstance(function, NodeOperation):
            function = Reduction(function)
        return function(self, axis=axis, *args, **kwargs)


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


class Branch(Node, collections.abc.Mapping):
    def __init__(self, *_children, _layer, **kwargs):
        super().__init__(**kwargs)
        self._layer = _layer
        self._children = _children
        for child in self._children:
            if self._layer not in child.metadata:
                raise ValueError(f"Missing '{self._layer}' attribute in layer item")
            child._parent = self

    def copy(self, _new_children=None, **kwargs):
        new = super().copy(**kwargs)
        if _new_children is None:
            _new_children = [child.copy() for child in self._children]
        new._layer = self._layer
        new._children = tuple(_new_children)
        for child in new._children:
            child._parent = new
        return new

    def __len__(self):
        return len(self._children)

    def __iter__(self):
        for child in self._children:
            yield child.metadata[self._layer]

    def __getitem__(self, key):
        # if key is a time window, we should make a new container by restricting to the time window at the data layer
        # if key is one of the keys for one of the items, return it
        # else, make a new container with asking each item for the key
        for child in self._children:
            if str(child.metadata[self._layer]) == str(key):
                return child
        # TODO: should probably restrict this to getting time windows. We need to access the time window class from here?
        # The logic for selecting a subset of a particular layer requires transferring
        # metadata, names, layers etc, which will probably not behave as intended without
        # some advanced logic. Instead, make a "select(layer, key)" method, which "reduces"
        # along that layer by selecting the nodes with the appropriate keys.
        children = [child[key] for child in self._children]
        return self.copy(_new_children=children)

    # @property
    # def _leaves(self):
    #     yield from itertools.chain(*[child._leaves for child in self._children])

    @property
    def _leaf_type(self):
        return self._children[0]._leaf_type

    # @property
    # def _nodes(self):
    #     for child in self._children:
    #         yield from child._nodes
    #     yield self

    def _traverse(self, leaves=True, branches=True, root=True, topdown=True, return_depth=False):
        if return_depth is True:
            return_depth = 1
        if root and topdown:
            yield self if not return_depth else (return_depth - 1, self)
        for child in self._children:
            yield from child._traverse(leaves=leaves, branches=branches, root=branches, topdown=topdown, return_depth=return_depth and return_depth + 1)
        if root and not topdown:
            yield self if not return_depth else (return_depth - 1, self)

    # @property
    # def _parent(self):
    #     return super()._parent

    # @_parent.setter
    # def _parent(self, parent):
    #     super(Branch, type(self))._parent.fset(self, parent)
    #     for child in self._children:
    #         child._parent = self  # Updates the list of metadata in the children

    # def __getattr__(self, name):
    #     values = [getattr(child, name) for child in self._children]
    #     if all([value == values[0] for value in values]):
    #         return values[0]
    #     raise AttributeError(f"'{self.__class__.__name__}' object has no single value for attribute '{name}'")


class NodeOperation:
    def __init__(self, function):
        self.function = function

    def __call__(self, node, *args, **kwargs):
        if isinstance(node, Branch):
            children = [self(child, *args, **kwargs) for child in node._children]
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
    def __call__(self, root, axis, metadata_merger='keep equal', *args, **kwargs):
        # TODO: add something that allows these reductions to be used on reducing axes on leaves as well?
        if isinstance(root, Leaf):
            # TODO: if we want to use the same reduction function for reducing over "layers" in the tree but also reducing over the axes of a data leaf, we will have problems with the axes somewhere.
            return root.reduce(self.function, axis=axis, *args, **kwargs)


        if axis != root._layer:
            return super().__call__(root, axis, *args, metadata_merger=metadata_merger, **kwargs)

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

        new = root._children[0].copy()
        for new_node, *old_nodes in zip(
                new._traverse(leaves=True, branches=True, root=True, topdown=False, return_depth=False),
                *(
                    child._traverse(leaves=True, branches=True, root=True, topdown=False, return_depth=False)
                    for child in root._children
                )
        ):
            if isinstance(new_node, Leaf):
                stacked = [item.data for item in old_nodes]
                try:
                    data = self.function(stacked, *args, **kwargs, axis=0)
                except TypeError as err:
                    if "got an unexpected keyword argument 'axis'" not in err:
                        raise
                    data = self.function(stacked, *args, **kwargs)
                new_node._data = data
            metadata = {}
            for key in new_node.metadata:
                stacked = [item.metadata[key] for item in old_nodes]
                labels = list(root)
                this_meta = metadata_merger(key, labels, stacked)
                if this_meta is not None:
                    metadata[key] = this_meta
                # TODO: implement different metadata merging strategies
                # - keep equal
                # - stack all
                # - mean
                # - custom function
                # Allow the user to customize this per metadata!
                # Probably by passing a dict with either names of pre-defined merging strategies,
                # or by passing a function that does the merging.
                # if all(item == value for item in stacked):
                    # metadata[key] = value
            new_node._metadata = metadata
        new._metadata = root._metadata | new._metadata  # This merge will promote metadata which survived the normal pruning, prioritizing the promoted data
            new_node.metadata = Metadata(node=new_node, **metadata)
        new.metadata = root.metadata | new.metadata  # This merge will promote metadata which survived the normal pruning, prioritizing the promoted data
        new.metadata.node = new
        return new


        # New is a mix of children and self. New will have children of the same structure as
        # self.children, which is why we copy one of the children to get a basic form.
        # New should have the same name as self, since we replace self with the reduction
        # of self's children.
        # New will have the same layer as self's children
        # New will also have the same metadata as self. Any metadata which was stored in
        # the children cannot be guaranteed to be consistent, and is therefore thrown away.
        # TODO: implement the above transfer of metadata.

        #
        # for newdata, *olddata in zip(new._leaves, *[child._leaves for child in self._children]):
        #     stacked = [item.data for item in olddata]
        #     newdata._data = func(stacked, *args, **kwargs)
        # return new


