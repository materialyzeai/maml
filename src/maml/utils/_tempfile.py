"""
Temporary directory and file creation utilities.
This file is adapted from monty.tempfile.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Any

from monty.shutil import copy_r, remove


class MultiScratchDir:
    """
    Automatically handles creation of temporary directories (utilizing Python's build in temp directory functions).

    The main difference between this class and monty ScratchDir is that multiple temp directories are created here.
    It enables the running of multiple jobs simultaneously in the directories
    The way it works is as follows:

    1. Create multiple temp dirs in specified root path.
    2. Optionally copy input files from current directory to temp dir.
    3. User loops among all directories
    4. User performs specified operations in each directories
    5. Change back to original directory.
    6. Delete temp dir.
    """

    SCR_LINK = "scratch_link"

    def __init__(
        self,
        rootpath: str | Path,
        n_dirs: int = 1,
        create_symbolic_link: bool = False,
        copy_from_current_on_enter: bool = False,
        copy_to_current_on_exit: bool = False,
    ):
        """
        Initializes scratch directory given a **root** path. There is no need
        to try to create unique directory names. The code will generate a
        temporary sub directory in the rootpath. The way to use this is using a
        with context manager. Example::
            with ScratchDir("/scratch"):
                do_something()
        If the root path does not exist or is None, this will function as a
        simple pass through, i.e., nothing happens.

        Args:
            rootpath (str/Path): Path in which to create temp subdirectories.
                If this is None, no temp directories will be created and
                this will just be a simple pass through.
            n_dirs (int): number of temporary directories to create
            create_symbolic_link (bool): Whether to create a symbolic link in
                the current working directory to the scratch directory
                created.
            copy_from_current_on_enter (bool): Whether to copy all files from
                the current directory (recursively) to the temp dir at the
                start, e.g., if input files are needed for performing some
                actions. Defaults to False.
            copy_to_current_on_exit (bool): Whether to copy files from the
                scratch to the current directory (recursively) at the end. E
                .g., if output files are generated during the operation.
                Defaults to False.
        """
        if Path is not None and isinstance(rootpath, Path):
            rootpath = str(rootpath)

        self.rootpath = os.path.abspath(rootpath) if rootpath is not None else None
        self.n_dirs = n_dirs
        self.cwd = os.getcwd()
        self.create_symbolic_link = create_symbolic_link
        self.start_copy = copy_from_current_on_enter
        self.end_copy = copy_to_current_on_exit
        self.tempdirs: list[str] = []

    def __enter__(self):
        tempdirs = [self.cwd] * self.n_dirs
        if self.rootpath is not None and os.path.exists(self.rootpath):
            tempdirs = [tempfile.mkdtemp(dir=self.rootpath) for _ in range(self.n_dirs)]
            self.tempdirs = [os.path.abspath(tempdir) for tempdir in tempdirs]
            if self.start_copy:
                [copy_r(".", tempdir) for tempdir in tempdirs]
            if self.create_symbolic_link:
                [os.symlink(tempdir, f"{MultiScratchDir.SCR_LINK}_{i}") for i, tempdir in enumerate(tempdirs)]
        return tempdirs

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object):
        if self.rootpath is not None and os.path.exists(self.rootpath):
            if self.end_copy:
                # First copy files over
                [_copy_r_with_suffix(tempdir, self.cwd, i) for i, tempdir in enumerate(self.tempdirs)]

            os.chdir(self.cwd)
            for tempdir in self.tempdirs:
                remove(tempdir)


def _copy_r_with_suffix(src: str, dst: str, suffix: Any | None = None):
    """
    Implements a recursive copy function similar to Unix's "cp -r" command.
    Surprisingly, python does not have a real equivalent. shutil.copytree
    only works if the destination directory is not present.

    Args:
        src (str): Source folder to copy.
        dst (str): Destination folder.
        suffix: Suffix to be added for copied files.
    """
    abssrc = os.path.abspath(src)
    absdst = os.path.abspath(dst)

    try:
        os.makedirs(absdst)
    except OSError:
        # If absdst exists, an OSError is raised. We ignore this error.
        pass
    for f in os.listdir(abssrc):
        fpath = os.path.join(abssrc, f)
        if os.path.isfile(fpath):
            if suffix is not None:
                new_path = f"{fpath}_{suffix!s}"
                shutil.copy(fpath, new_path)
                fpath = new_path
            shutil.copy(fpath, absdst)

        elif not absdst.startswith(fpath):
            _copy_r_with_suffix(fpath, os.path.join(absdst, f), suffix=suffix)
        else:
            warnings.warn(f"Cannot copy {fpath} to itself")
