# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Vendored build output must be copyable into a portable source checkout."""

import os
import shutil
import subprocess
from pathlib import Path


def test_deepgemm_build_output_can_populate_source_checkout(tmp_path):
    source = Path(__file__).resolve().parents[2] / "vllm/third_party/deep_gemm"
    destination = tmp_path / "deep_gemm"
    if source.is_symlink() and Path(os.readlink(source)).is_absolute():
        # A clean builder has no contributor-local checkout at the link target.
        destination.symlink_to(tmp_path / "absent-author-checkout/deep_gemm")
    built = tmp_path / "build/deep_gemm"
    built.mkdir(parents=True)
    (built / "__init__.py").write_text("# Vendored build output\n")
    shutil.copytree(built, destination, dirs_exist_ok=True)
    assert (destination / "__init__.py").read_bytes() == (
        built / "__init__.py"
    ).read_bytes()


def test_dependency_recipe_keys_mutable_fetch_and_native_caches(tmp_path):
    """Python-only edits reuse cache; dependency patches select separate storage."""
    root = Path(__file__).resolve().parents[2]
    builder = (root / "tools/jovian_wheel_release/build_bundle.sh").read_text()
    dockerfile = (root / "tools/jovian_wheel_release/Dockerfile").read_text()
    assert "rev-parse HEAD:cmake/external_projects" in builder
    assert '--build-arg "DEPENDENCY_RECIPE=${dependency_recipe}"' in builder
    for cache in (
        "native-cu134-torch214-sm120",
        "fetchcontent-cu134",
        "generated-cu134-torch214-sm120",
    ):
        assert f"id=lil-vllm-{cache}-${{DEPENDENCY_RECIPE}}," in dockerfile

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(tmp_path), *args], text=True
        ).strip()

    git("init", "--quiet")
    git("config", "user.name", "Cache contract test")
    git("config", "user.email", "cache@example.invalid")
    git("config", "commit.gpgsign", "false")
    recipes = tmp_path / "cmake/external_projects"
    recipes.mkdir(parents=True)
    (recipes / "flashkda.cmake").write_text("dependency recipe A\n")
    git("add", ".")
    git("commit", "--quiet", "-m", "Dependency fixture")
    identity = git("rev-parse", "HEAD:cmake/external_projects")
    (tmp_path / "model.py").write_text("# Independent Python serving source\n")
    git("add", ".")
    git("commit", "--quiet", "-m", "Python fixture")
    assert git("rev-parse", "HEAD:cmake/external_projects") == identity
    (recipes / "flashkda.patch").write_text("tracked dependency patch\n")
    git("add", ".")
    git("commit", "--quiet", "-m", "Patched dependency fixture")
    assert git("rev-parse", "HEAD:cmake/external_projects") != identity
