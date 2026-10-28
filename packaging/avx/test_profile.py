#!/usr/bin/env python3
"""Focused build-profile checks; no dependency downloads or Paddle import."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from audit_compile_commands import REQUIRED, audit, violations

ROOT = Path(__file__).resolve().parents[2]
CMAKE = shutil.which("cmake")


class CompilerAuditTests(unittest.TestCase):
    def test_known_baseline_and_argument_arrays(self):
        record = dict(file="example.cc", arguments=["c++", *sorted(REQUIRED), "-c", "example.cc"])
        self.assertEqual(audit([record]), [])

    def test_stronger_flags_are_rejected_even_after_baseline(self):
        for flag in ("-mavx2", "-mfma", "-mfma4", "-mavx512bw", "-march=native", "-march=x86-64-v3"):
            with self.subTest(flag=flag):
                self.assertTrue(violations(dict(command="c++ " + " ".join(sorted(REQUIRED)) + " " + flag)))

    def test_missing_baseline_and_empty_inventory_are_rejected(self):
        self.assertTrue(violations(dict(command="c++ -c example.cc")))
        with self.assertRaises(ValueError):
            audit([])


@unittest.skipUnless(CMAKE, "CMake is required for profile checks")
class CMakeProfileTests(unittest.TestCase):
    def test_cache_sets_cpu_backend_and_disables_newer_isa(self):
        with tempfile.TemporaryDirectory(prefix="avx-profile-") as directory:
            script = Path(directory) / "check.cmake"
            script.write_text('include("' + (ROOT / "cmake/avx_cpu.cmake").as_posix() + '")\n' +
                'foreach(flag PADDLE_AVX_CPU WITH_AVX AVX_FOUND WITH_POCKETFFT)\n'
                '  if(NOT ${flag})\n    message(FATAL_ERROR "Expected ${flag}=ON")\n  endif()\nendforeach()\n'
                'foreach(flag WITH_GPU WITH_ROCM WITH_MKL WITH_ONEMKL WITH_ONEDNN AVX2_FOUND AVX512F_FOUND)\n'
                '  if(${flag})\n    message(FATAL_ERROR "Expected ${flag}=OFF")\n  endif()\nendforeach()\n'
                'if(NOT CMAKE_C_FLAGS STREQUAL CMAKE_CXX_FLAGS)\n  message(FATAL_ERROR "Different C/C++ ISA flags")\nendif()\n'
                'file(WRITE "${CMAKE_CURRENT_LIST_DIR}/flags.txt" "${CMAKE_CXX_FLAGS}")\n')
            subprocess.run([CMAKE, "-P", str(script)], check=True, capture_output=True, text=True)
            flags = (Path(directory) / "flags.txt").read_text()
            self.assertEqual(violations(dict(command="c++ " + flags)), [])

    def test_openblas_profile_and_upstream_default(self):
        with tempfile.TemporaryDirectory(prefix="avx-openblas-") as directory:
            root = Path(directory)
            dependency = root / "third_party/openblas"
            dependency.mkdir(parents=True)
            (dependency / "README").write_text("Invented source placeholder; no download.\n")
            fake_git = root / "git-version"
            fake_git.write_text("#!/bin/sh\nprintf '%s\\n' v0.3.28\n")
            fake_git.chmod(0o755)
            script = root / "CMakeLists.txt"
            script.write_text('cmake_minimum_required(VERSION 3.18)\nproject(profile C)\n'
                'set(PADDLE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")\n'
                'set(THIRD_PARTY_PATH "${CMAKE_BINARY_DIR}/third_party")\n'
                'set(GIT_EXECUTABLE "' + fake_git.as_posix() + '")\nset(NPROC 1)\n'
                'include("' + (ROOT / "cmake/external/openblas.cmake").as_posix() + '")\n'
                'get_target_property(command extern_openblas _EP_BUILD_COMMAND)\n'
                'file(WRITE "${CMAKE_BINARY_DIR}/command.txt" "${command}")\n')
            for enabled in (True, False):
                build = root / ("avx" if enabled else "default")
                subprocess.run([CMAKE, "-S", str(root), "-B", str(build),
                                "-DPADDLE_AVX_CPU=" + ("ON" if enabled else "OFF")],
                               check=True, capture_output=True, text=True)
                command = (build / "command.txt").read_text()
                self.assertIn("DYNAMIC_ARCH=" + ("0" if enabled else "1"), command)
                self.assertIn("NUM_THREADS=" + ("1" if enabled else "64"), command)
                if enabled:
                    self.assertIn("TARGET=SANDYBRIDGE", command)
                    self.assertIn("NOFORTRAN=1", command)
                else:
                    self.assertNotIn("TARGET=SANDYBRIDGE", command)


if __name__ == "__main__":
    unittest.main(verbosity=2)
