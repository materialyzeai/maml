from __future__ import annotations

import os
import unittest
from glob import glob
from pathlib import Path

import pytest
from monty.tempfile import ScratchDir

from maml.utils._tempfile import MultiScratchDir, _copy_r_with_suffix


class TestMultiScratchDir(unittest.TestCase):
    def test_multi(self):
        with ScratchDir("."):
            with MultiScratchDir(".", n_dirs=4, copy_to_current_on_exit=True) as dirs:
                assert len(dirs) == 4
                for d in dirs:
                    os.chdir(d)
                    os.system("touch tempfile")
                    os.chdir("..")
            tempfiles = set(glob("tempfile*"))
            assert {("tempfile_%d" % i) for i in range(4)} == tempfiles


def test_multi_scratch_dir_copy_and_symlink(tmp_path, monkeypatch):
    workdir = tmp_path / "work"
    workdir.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (workdir / "input.txt").write_text("hello")
    monkeypatch.chdir(workdir)
    with MultiScratchDir(scratch, n_dirs=2, create_symbolic_link=True, copy_from_current_on_enter=True) as dirs:
        assert len(dirs) == 2
        for i, d in enumerate(dirs):
            assert os.path.dirname(d) == str(scratch.resolve())
            assert (Path(d) / "input.txt").read_text() == "hello"
            assert os.path.realpath(f"{MultiScratchDir.SCR_LINK}_{i}") == os.path.realpath(d)
    assert os.getcwd() == str(workdir)
    assert not any(Path(d).exists() for d in dirs)


def test_multi_scratch_dir_missing_root_is_passthrough(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with MultiScratchDir(tmp_path / "does_not_exist", n_dirs=3) as dirs:
        assert dirs == [str(tmp_path)] * 3
    assert tmp_path.exists()


def test_copy_r_with_suffix_recursive(tmp_path):
    src = tmp_path / "src"
    (src / "sub").mkdir(parents=True)
    (src / "a.txt").write_text("a")
    (src / "sub" / "b.txt").write_text("b")
    dst = tmp_path / "dst"
    _copy_r_with_suffix(str(src), str(dst), suffix=1)
    assert (dst / "a.txt_1").read_text() == "a"
    assert (dst / "sub" / "b.txt_1").read_text() == "b"

    dst2 = tmp_path / "dst2"
    _copy_r_with_suffix(str(src), str(dst2))
    assert (dst2 / "a.txt").read_text() == "a"


def test_copy_r_with_suffix_into_itself_warns(tmp_path):
    src = tmp_path / "src"
    (src / "inner").mkdir(parents=True)
    with pytest.warns(UserWarning, match="Cannot copy"):
        _copy_r_with_suffix(str(src), str(src / "inner"))


if __name__ == "__main__":
    unittest.main()
