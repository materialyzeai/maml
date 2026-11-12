"""
Pyinvoke tasks.py file for automating releases and admin stuff.

Run with the maintain dependency group, which has invoke and the Sphinx tools:

    uv run --group maintain invoke release

Author: Shyue Ping Ong
"""

from __future__ import annotations

import datetime as dt
import glob
import re
import tempfile
from contextlib import chdir
from pathlib import Path

from invoke import task

REPO = "materialyzeai/maml"


@task
def make_docs(ctx):
    """
    Generate API documentation + run Sphinx.

    Args:
        ctx: Invoke context.
    """
    with chdir("docs"):
        ctx.run("rm -f maml.*.rst")
        ctx.run("touch index.rst")
        ctx.run("sphinx-apidoc -P -M -d 6 -o . -f ../src/maml")
        ctx.run("rm -f maml*.tests.*rst")
        ctx.run("sphinx-build -M markdown . .")
        ctx.run("rm -f *.rst")
        ctx.run("cp markdown/maml*.md .")
        for fn in glob.glob("maml*.md"):
            with open(fn) as f:
                lines = [line.rstrip() for line in f if "Submodules" not in line]
            if fn == "maml.md":
                preamble = ["---", "layout: default", "title: API Documentation", "nav_order: 5", "---", ""]
            else:
                preamble = ["---", "layout: default", "title: " + fn, "nav_exclude: true", "---", ""]
            with open(fn, "w") as f:
                f.write("\n".join(preamble + lines))

        ctx.run("rm -rf markdown")
        ctx.run("cp ../*.md .")
        ctx.run("mv README.md index.md")
        ctx.run("rm -rf *.orig doctrees")

        with open("index.md") as f:
            contents = f.read()
        with open("index.md", "w") as f:
            contents = re.sub(
                r"\n## Official Documentation[^#]*",
                "{: .no_toc }\n\n## Table of contents\n{: .no_toc .text-delta }\n* TOC\n{:toc}\n\n",
                contents,
            )
            contents = "---\nlayout: default\ntitle: Home\nnav_order: 1\n---\n\n" + contents

            f.write(contents)


@task
def update_docs(ctx):
    """
    Regenerate the docs and push them.

    Args:
        ctx: Invoke context.
    """
    make_docs(ctx)
    ctx.run("git add docs")
    ctx.run('git commit -m "Update docs"')
    ctx.run("git push")


@task
def publish(ctx):
    """
    Build and upload to PyPI from this machine. Normally not needed: publishing a GitHub release
    (see `release`) makes CI test, build and publish to PyPI with trusted publishing.

    Args:
        ctx: Invoke context.
    """
    ctx.run("rm -rf dist")
    ctx.run("uv build")
    ctx.run("uv publish")


@task
def set_ver(ctx, version):
    """
    Set the version in pyproject.toml and update uv.lock.

    Args:
        ctx: Invoke context.
        version (str): New version, e.g. "2026.11.12".
    """
    path = Path("pyproject.toml")
    text, count = re.subn(r'^version = "[^"]*"', f'version = "{version}"', path.read_text(), count=1, flags=re.M)
    if count != 1:
        raise ValueError("No version line found in pyproject.toml.")
    path.write_text(text)
    ctx.run("uv lock")


def release_notes(version: str, changelog: str = "CHANGES.md") -> str:
    """
    Get the release notes for a version from the changelog.

    Args:
        version (str): Version, e.g. "2026.11.12".
        changelog (str): Changelog with one "## v<version>" (or "## <version>") section per release.

    Returns:
        str: The body of the section.

    Raises:
        ValueError: If the changelog has no section for the version.
    """
    text = Path(changelog).read_text()
    match = re.search(rf"^## v?{re.escape(version)}[ \t]*\n(.*?)(?=^## |\Z)", text, flags=re.M | re.S)
    if not match or not match.group(1).strip():
        raise ValueError(f"{changelog} has no '## v{version}' section. Add the release notes first.")
    return match.group(1).strip()


@task
def release(ctx, version=None, notest=True, nodoc=False):
    """
    Release a new version: set the version, regenerate the docs, commit and push, then create the
    GitHub release. Publishing the release makes CI run the tests and publish to PyPI.

    Args:
        ctx: Invoke context.
        version (str): Version to release. Defaults to today's date, e.g. "2026.11.12".
        notest (bool): Skip running the tests locally first. CI runs them before publishing.
        nodoc (bool): Skip regenerating the docs.
    """
    version = version or dt.datetime.now().strftime("%Y.%-m.%-d")
    notes = release_notes(version)
    if not notest:
        ctx.run("uv run pytest tests")
    set_ver(ctx, version=version)
    if not nodoc:
        make_docs(ctx)
    ctx.run("git add pyproject.toml uv.lock CHANGES.md docs")
    ctx.run(f'git commit -m "Release v{version}"')
    ctx.run("git push")

    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(notes)
    ctx.run(
        f"gh release create v{version} --repo {REPO} --target main --title v{version} --notes-file {f.name}",
    )
    Path(f.name).unlink()
