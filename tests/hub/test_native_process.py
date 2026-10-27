"""CPU-only process-completion fixtures. Never register these with Hub."""
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from hub_runtime.native_process import NativeProcess


@unittest.skipUnless(sys.platform.startswith('linux'), 'Linux /proc completion evidence required')
class NativeCompletionTest(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-00000000-0000-0000-0000-000000000000'})
        self.env.start()
        self.directory = tempfile.TemporaryDirectory()
        self.log = str(Path(self.directory.name) / 'native.log')

    def tearDown(self):
        self.env.stop()
        self.directory.cleanup()

    def test_gpu_assignment_reaches_native_child_and_real_exit_is_complete(self):
        model = NativeProcess()
        result = model.run([sys.executable, '-c', 'import os; print(os.environ["CUDA_VISIBLE_DEVICES"]); print(os.environ["AUDIO_GPU_UUID"])'], log_path=self.log)
        self.assertEqual(result['returncode'], 0)
        self.assertEqual(Path(self.log).read_text().splitlines(), [os.environ['CUDA_VISIBLE_DEVICES']] * 2)
        model.complete()
        self.assertFalse(model.groups)

    def test_native_failure_is_not_a_completion_uncertainty(self):
        model = NativeProcess()
        result = model.run([sys.executable, '-c', 'raise SystemExit(7)'], log_path=self.log)
        self.assertEqual(result['returncode'], 7)
        model.complete()

    def test_timeout_preserves_running_process_and_cannot_report_completion(self):
        model = NativeProcess()
        with self.assertRaises(TimeoutError):
            model.run([sys.executable, '-c', 'import time; time.sleep(.25)'], timeout=.02, log_path=self.log)
        self.assertTrue(any(process.poll() is None for process in model.processes.values()))
        with self.assertRaises(RuntimeError):
            model.complete()
        time.sleep(.3)
        model.complete()

    def test_parent_exit_does_not_hide_live_native_descendants(self):
        model = NativeProcess()
        child = 'import time; time.sleep(.25)'
        parent = 'import subprocess,sys; subprocess.Popen([sys.executable,"-c",'+repr(child)+'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)'
        with self.assertRaisesRegex(RuntimeError, 'descendants'):
            model.run([sys.executable, '-c', parent], log_path=self.log)
        with self.assertRaises(RuntimeError):
            model.complete()
        time.sleep(.3)
        model.complete()

    def test_native_worker_does_not_receive_hub_machine_credentials(self):
        with patch.dict(os.environ, {'HUB_RUNTIME_TOKEN': 'cpu-fixture-not-a-credential'}):
            model = NativeProcess()
        model.run([sys.executable, '-c', 'import os; print(os.environ.get("HUB_RUNTIME_TOKEN", "absent"))'], log_path=self.log)
        self.assertEqual(Path(self.log).read_text().strip(), 'absent')
        model.complete()

    def test_http_parent_cannot_create_an_unassigned_gpu_runner(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}):
            with self.assertRaises(RuntimeError):
                NativeProcess()


if __name__ == '__main__':
    unittest.main()
