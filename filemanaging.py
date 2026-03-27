#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 15:03:07 2020

@author: robin
"""

import os
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt


def get_unique_fname(basename, modifier=None):
    """
    Check if a file exists and if yes return a unique file name.

    Parameters
    ----------
    basename : str
        Name of the file that will return if there is not already a file with
        this name
    modifier : function, optional
        Function that accepts a positive integer representing the number of
        tries (starting at 1) generating a new name. It is appended to the
        basename in order to generate the unique file name. The default appends
        '_copy_i' where i is the argument of modifier.

    """
    if os.path.isfile(basename):
        if modifier is None:
            modifier = lambda i: '_copy' if i == 1 else f'_copy_{i}'
        dotpos = basename.rfind('.')
        if dotpos == -1:
            dotpos = len(basename)
        extension = basename[dotpos:]
        noextension = basename[:dotpos]
        i = 1
        while True:
            newname = noextension + modifier(i) + extension
            i += 1
            if not os.path.isfile(newname):
                return newname
    return basename


def get_unique_dname(basename, modifier=None):
    """
    Generate an unique directory name and return it.

    Parameters
    ----------
    basename : str
        Name of the directory that will be used if no other directory exists
        with the same name.
    modifier : function, optional
        See `get_unique_fname`.

    """
    if not os.path.isdir(basename):
        return basename
    if modifier is None:
        modifier = lambda i: '_copy' if i == 1 else f'_copy_{i}'
    i = 1
    while True:
        newname = basename + modifier(i)
        i += 1
        if not os.path.isdir(newname):
            return newname


def create_unique_dir(basename, modifier=None):
    """
    Create new directory and return its name.

    Parameters
    ----------
    basename : str
        Name of the directory that will be used if no other directory exists
        with the same name.
    modifier : function, optional
        See `get_unique_fname`.

    """
    dname = get_unique_dname(basename, modifier)
    os.makedirs(dname)
    return dname


def to_keyvalstr(keyvals):
    """
    Convert series of key-value pairs into a string.

    Parameters
    ----------
    keyvals : dict
        Dictionary of key-value pairs to be saved as a string.

    """
    out = ''
    for key, val in keyvals.items():
        if isinstance(val, (int, float, str, complex)):
            outval = str(val)
        elif isinstance(val, (list, tuple, np.ndarray)):
            outval = '_'.join(map(str, val))
        else:
            raise ValueError(f'unsupported value typy for key {key}')
        out += f'{key}_{outval}'
    return out


class PersistentDataFrame(pd.DataFrame):
    """
    DataFrame class that synchronizes with a csv file and monitors changes.
    """

    @classmethod
    def from_file(cls, fname):
        df = pd.read_csv(index_col=0)
        return cls(fname, df)

    def __init__(self, fname, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fname = fname
        self.save(force=True)

    def get_lastmod_file(self):
        """Check when csv file was modified."""
        timestamp = os.path.getmtime(self.fname)
        dt = datetime.datetime.fromtimestamp(timestamp)
        return dt

    def update_last_sync(self):
        """Set last sync time to last file modification time."""
        self.last_sync = self.get_lastmod_file()

    def save(self, force=False):
        """
        Save DataFrame to csv file.

        Parameters
        ----------
        force : bool, optional
            Determines whether file is overwritten if it has been changed after
            the last synchronization.
        """
        if (not force) and (self.last_sync < self.get_lastmod_file()):
            raise RuntimeError('csv file has been changed since last sync')
        self.to_csv(self.fname)
        self.update_last_sync()

    def load(self):
        """Load data from csv file and return new DataFrame."""
        return self.from_file(self.fname)


class DataFramePlotter:
    """
    Class that allows to select x and y axes from columns of a DataFrame and
    plots certain rows based on constraints for other axes (parameters).

    Contstraints:
        Filter data based, e.g., on equalities, inequalities
    """

    def __init__(self, df, xcol, ycol, opts=None):
        self.df = df
        self.xcol = xcol
        self.ycol = ycol
        if opts is None:
            opts = dict()
        if 'pltopts' not in opts:
            opts['pltopts'] = dict()
        if 'xlabel' not in opts:
            opts['xlabel'] = ''
        if 'ylabel' not in opts:
            opts['ylabel'] = ''
        self.opts = opts

    def plot(self, colconstrs=dict()):
        df = self.df
        xcol = self.xcol
        ycol = self.ycol
        pltopts = self.opts['pltopts']
        xlabel = self.opts['xlabel']
        ylabel = self.opts['ylabel']
        fig = plt.figure()
        ax = fig.gca()
        xs = df[xcol]
        ys = df[ycol]
        ax.plot(xs, ys, **pltopts)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return fig

