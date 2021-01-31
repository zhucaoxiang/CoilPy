import h5py
import os  # for path.abspath
import keyword  # for getting python keywords

# Modified from J. Schilling (jonathan.schilling@ipp.mpg.de)'s script for SPEC


class HDF5:
    """Create a python object for a HDF5 file.

    Returns:
        HDF5 class: python HDF5 object.

    Use as s = HDF5(filename), e.g. s=HDF5("ext.h5")
    This class can be iterated or entered.
    To check all the items, you can use `self.inventory()`.
    """

    def __init__(self, *args, **kwargs):
        """Constructor

        Args:
            arg[0] (str): The name of a file or an item inside the root object.
                          If args[0] is not a filename, kwargs['content'] should be the
                          content to be added as self.`args[0]`.
        """

        _content = None
        if kwargs.get("content") is None:
            # assume arg[0] is a filename
            _content = h5py.File(args[0], "r")

            # keep track of which file this object corresponds to
            self.filename = os.path.abspath(args[0])
        elif isinstance(kwargs["content"], h5py.Group):
            _content = kwargs["content"]

        if _content is not None:
            for key in _content:
                if isinstance(_content[key], h5py.Group):
                    # recurse into group
                    setattr(self, key, HDF5(content=_content[key]))
                elif isinstance(_content[key], h5py.Dataset):  # read dataset
                    if (
                        key in keyword.kwlist
                    ):  # add underscore avoiding assigning python keywords
                        setattr(self, key + "_", _content[key][()])
                    else:
                        try:  # this should be simplified when FOCUS writes the correct format
                            if len(_content[key][()]) == 1:
                                # if just one element, use the value directly
                                setattr(self, key, _content[key][0])
                            else:  # arrays
                                setattr(self, key, _content[key][()])
                        except TypeError:  # scalar
                            setattr(self, key, _content[key][()])

        if isinstance(_content, h5py.File):
            _content.close()

    # needed for iterating over the contents of the file
    def __iter__(self):
        return iter(self.__dict__)

    def __next__(self):
        return next(self.__dict__)

    def __enter__(self):
        return self

    def __exit__(self, t, v, tb):
        return

    def inventory(self, prefix=""):
        """Print a list of items contained in this object

        Args:
            prefix (str, optional): Header to be printed. Defaults to "".
        """
        _prefix = ""
        if prefix != "":
            _prefix = prefix + "/"

        for a in self:
            try:
                # recurse into member
                getattr(self, a).inventory(prefix=_prefix + a)
            except AttributeError:
                # print item name
                print(_prefix + a)
