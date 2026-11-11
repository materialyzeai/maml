"""
Pyinvoke tasks.py file for automating releases and admin stuff.

Author: Shyue Ping Ong
"""

from __future__ import annotations

import datetime
import glob
import json
import os
import re

import requests
from invoke import task
from monty.os import cd


@task
def make_docs(ctx):
    """
    Generate API documentation + run Sphinx.

    :param ctx:
    """

    with cd("docs"):
        ctx.run("rm maml.*.rst", warn=True)
        ctx.run("touch index.rst", warn=True)
        ctx.run("sphinx-apidoc -P -M -d 6 -o . -f ../src/maml")
        ctx.run("rm maml*.tests.*rst", warn=True)
        ctx.run("sphinx-build -M markdown . .")
        ctx.run("rm *.rst", warn=True)
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

        ctx.run("rm -r markdown", warn=True)
        ctx.run("cp ../*.md .")
        ctx.run("mv README.md index.md")
        ctx.run("rm -rf *.orig doctrees", warn=True)

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
    Update the web documentation.

    :param ctx:
    """
    ctx.run("cp README.rst docs/")
    ctx.run("cp api-docs-source/conf.py docs/conf.py")
    make_docs(ctx)
    ctx.run("git add .")
    ctx.run('git commit -a -m "Update docs"')
    ctx.run("git push")


@task
def publish(ctx):
    """
    Upload release to Pypi using twine.

    :param ctx:
    """
    ctx.run("rm dist/*.*", warn=True)
    ctx.run("python -m build")
    ctx.run("twine upload dist/*")


@task
def set_ver(ctx, version):
    lines = []
    with open("pyproject.toml") as f:
        for l in f:
            lines.append(re.sub(r"^version = ([^,]+)", f'version = "{version}"', l.rstrip()))
    with open("pyproject.toml", "w") as f:
        f.write("\n".join(lines) + "\n")


@task
def release(ctx, version=datetime.datetime.now().strftime("%Y.%-m.%-d"), notest=True, nodoc=False):
    set_ver(ctx, version=version)
    ctx.run("rm -r dist build maml.egg-info", warn=True)
    if not notest:
        ctx.run("pytest maml")
    with open("CHANGES.md") as f:
        contents = f.read()
    toks = re.split(r"\-+", contents)
    desc = toks[1].strip()
    toks = desc.split("\n")
    desc = "\n".join(toks[:-1]).strip()
    if not nodoc:
        make_docs(ctx)
        ctx.run("git add .")
        ctx.run('git commit -a -m "Update docs"')
        ctx.run("git push")

    payload = {
        "tag_name": "v" + version,
        "target_commitish": "master",
        "name": "v" + version,
        "body": desc,
        "draft": False,
        "prerelease": False,
    }
    response = requests.post(
        "https://api.github.com/repos/materialsvirtuallab/maml/releases",
        data=json.dumps(payload),
        headers={"Authorization": "token " + os.environ["GITHUB_RELEASES_TOKEN"]},
    )
    print(response.text)
    ctx.run("rm -f dist/*.*", warn=True)
    ctx.run("python -m build", warn=True)
    ctx.run("twine upload --skip-existing dist/*.whl", warn=True)
    ctx.run("twine upload --skip-existing dist/*.tar.gz", warn=True)
