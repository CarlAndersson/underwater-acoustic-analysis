import numpy as np
import collections.abc


class Node:
    def __init__(self, _metadata=None):
        self._metadata = _metadata or {}
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
        obj._metadata = self._metadata.copy()
        obj._parent = None
        return obj

    def __getattr__(self, name):
        if name in ('_metadata', '_parent_metadata', '_parent'):
            # Safeguard for infinite recursion trying to get these attributes.
            # This should only happen if the object is not properly initialized.
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        if name in self._metadata:
            # The requested attribute is stored in the metadata
            return self._metadata[name]
        if name in self._parent_metadata:
            # The requested attribute is stored by the parent, i.e. is the same for all the siblings
            return getattr(self._parent, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __dir__(self):
        # return list(self._metadata.keys()) + self._parent_metadata + super().__dir__()
        props = super().__dir__()
        props += list(self._metadata.keys())
        props += self._parent_metadata
        return props

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

    @property
    def _parent_metadata(self):
        if self._parent is None:
            return []
        return list(self._parent._metadata.keys()) + self._parent._parent_metadata + [self._parent._layer]
        # for child in self._children:
        #     child._parent = self  # Updates the list of metadata in the children



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

    def apply(self, func, apply_to_data=True, *args, **kwargs):
        if isinstance(self.data, Leaf):
            out = self.data.apply(func, apply_to_data=apply_to_data, *args, **kwargs)
        elif apply_to_data:
            out = func(self.data, *args, **kwargs)
        else:
            out = func(self, *args, **kwargs)

        if isinstance(out, Leaf):
            return out
        obj = self.copy()
        obj._data = out
        return obj


class Branch(Node, collections.abc.Mapping):
    def __init__(self, *_children, _layer, **kwargs):
        super().__init__(**kwargs)
        self._layer = _layer
        self._children = _children
        for _child in self._children:
            if self._layer not in _child._metadata:
                raise ValueError(f"Missing '{self._layer}' attribute in layer item")
            _child._parent = self

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
            yield child._metadata[self._layer]

    def __getitem__(self, key):
        # if key is a time window, we should make a new container by restricting to the time window at the data layer
        # if key is one of the keys for one of the items, return it
        # else, make a new container with asking each item for the key
        for child in self._children:
            if str(child._metadata[self._layer]) == str(key):
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

    def apply(self, func, *args, axis=None, metadata_merger='keep equal', apply_to_data=True, **kwargs):
        """

        Parameters
        ---------
        func : callable
            Function to apply to the data
        axis : named axis
            The axis over which to apply the function. The named axis will be reduced from the data.
        metadata_merger : {callable, 'keep equal', 'stack', 'collect'}
            How to transfer metadata when reducing an axis.
            Three default strategies are implemented:

                - 'keep equal': Only keeps the metadata values where all the items have the same value.
                - 'stack': Always stacks the metadata in a list.
                - 'collect': Collects the metadata in a dictionary indexed by the label/name of the item containing the metadata.

            If it is a callable, it should have the signature
                ``out = fun(name, labels, data)``
            where `name` is the name of the metadata, `labels` is a list of the labels of each item reduced over,
            and `data` is a list of the actual metadata.
            The function should return the new metadata. Returning `None` causes the metadata to be dropped.
            Note that this function will be called for all metadata below the layer where the reduction takes place.
        """
        if axis is None:
            children = [child.apply(func, *args, metadata_merger=metadata_merger, apply_to_data=apply_to_data, **kwargs) for child in self._children]
            return self.copy(_new_children=children)
            return type(self)(items, layer=self.layer, key=self.key)

        if axis != self._layer:
            children = [child.apply(func, *args, axis=axis, metadata_merger=metadata_merger, apply_to_data=apply_to_data, **kwargs) for child in self._children]
            return self.copy(_new_children=children)
            return type(self)(items, layer=self.layer, key=self.key)

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

        new = self._children[0].copy()
        for new_node, *old_nodes in zip(
                new._traverse(leaves=True, branches=True, root=True, topdown=False, return_depth=False),
                *(
                    child._traverse(leaves=True, branches=True, root=True, topdown=False, return_depth=False)
                    for child in self._children
                )
        ):
            if isinstance(new_node, Leaf):
                if apply_to_data:
                    stacked = [item.data for item in old_nodes]
                else:
                    stacked = [item for item in old_nodes]
                new_node._data = func(stacked, *args, **kwargs)
            metadata = {}
            for key in new_node._metadata:
                stacked = [item._metadata[key] for item in old_nodes]
                labels = [item._name for item in old_nodes]
                labels = list(self.keys())
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
        new._name = self._name
        new._metadata = new._metadata | self._metadata  # This merge will promote metadata which survived the normal pruning.
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


class Transform:
    def __call__(self, node, *args, **kwargs):
