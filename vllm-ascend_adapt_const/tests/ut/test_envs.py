#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

import inspect
import os

import vllm_ascend.envs as envs_ascend
from tests.ut.base import TestBase


class TestEnvVariables(TestBase):
    def setUp(self):
        self.env_vars = list(envs_ascend.env_variables.keys())

    def test_env_vars_behavior(self):
        for var_name in self.env_vars:
            with self.subTest(var=var_name):
                original_val = os.environ.get(var_name)
                var_handler = envs_ascend.env_variables[var_name]

                try:
                    if var_name in os.environ:
                        del os.environ[var_name]
                    self.assertEqual(getattr(envs_ascend, var_name), var_handler())

                    handler_source = inspect.getsource(var_handler)
                    if "int(" in handler_source:
                        test_vals = ["123", "456"]
                    elif "bool(int(" in handler_source:
                        test_vals = ["0", "1"]
                    else:
                        test_vals = [f"test_{var_name}", f"custom_{var_name}"]

                    for test_val in test_vals:
                        os.environ[var_name] = test_val
                        self.assertEqual(getattr(envs_ascend, var_name), var_handler())

                finally:
                    if original_val is None:
                        os.environ.pop(var_name, None)
                    else:
                        os.environ[var_name] = original_val

    def test_dir_and_getattr(self):
        self.assertEqual(sorted(envs_ascend.__dir__()), sorted(self.env_vars))
        for var_name in self.env_vars:
            with self.subTest(var=var_name):
                getattr(envs_ascend, var_name)

    def test_compute_aware_grouping_defaults_and_overrides(self):
        expected_defaults = {
            "VLLM_ASCEND_PP_COMPUTE_AWARE_MIN_TOKENS": 512,
            "VLLM_ASCEND_PP_COMPUTE_AWARE_MIN_GAIN_PCT": 5,
            "VLLM_ASCEND_PP_COMPUTE_AWARE_QUANTUM": 8,
            "VLLM_ASCEND_PP_SCOM_MIN_GAIN_PCT": 3,
            "VLLM_ASCEND_PP_SCOM_CAPACITY_QUANTUM": 64,
            "VLLM_ASCEND_PP_SCOM_MAX_CAPACITY_CANDIDATES": 8,
            "VLLM_ASCEND_PP_SCOM_MAX_SWAPS": 4,
        }
        originals = {
            name: os.environ.get(name) for name in expected_defaults
        }
        try:
            for name, expected in expected_defaults.items():
                os.environ.pop(name, None)
                self.assertEqual(
                    getattr(envs_ascend, name),
                    expected,
                )
                os.environ[name] = str(expected + 1)
                self.assertEqual(
                    getattr(envs_ascend, name),
                    expected + 1,
                )
        finally:
            for name, original in originals.items():
                if original is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = original

    def test_scom_grouping_defaults_and_overrides(self):
        expected_defaults = {
            "VLLM_ASCEND_PP_MICROBATCH_GROUPING": "scom",
            "VLLM_ASCEND_PP_SCOM_SHAPE_BUCKETS": (
                "128,256,512,1024,2048,4096,8192"
            ),
            "VLLM_ASCEND_PP_SCOM_OPTIMIZE_CAPACITIES": True,
            "VLLM_ASCEND_PP_SCOM_ALLOW_BUCKET_CROSSING": False,
        }
        originals = {
            name: os.environ.get(name) for name in expected_defaults
        }
        try:
            for name, expected in expected_defaults.items():
                os.environ.pop(name, None)
                self.assertEqual(getattr(envs_ascend, name), expected)
            os.environ[
                "VLLM_ASCEND_PP_SCOM_OPTIMIZE_CAPACITIES"
            ] = "0"
            os.environ[
                "VLLM_ASCEND_PP_SCOM_ALLOW_BUCKET_CROSSING"
            ] = "1"
            self.assertFalse(
                envs_ascend
                .VLLM_ASCEND_PP_SCOM_OPTIMIZE_CAPACITIES
            )
            self.assertTrue(
                envs_ascend
                .VLLM_ASCEND_PP_SCOM_ALLOW_BUCKET_CROSSING
            )
        finally:
            for name, original in originals.items():
                if original is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = original
