"""Completion barrier for the provider's existing per-job native CLI.

Hub SDK owns admission and residency. This helper only runs the unchanged native
command inside the assigned engine and proves its process group has completed.
"""
import gc
import os
from pathlib import Path
import subprocess


def members(group):
    found = []
    for directory in Path('/proc').iterdir():
        if not directory.name.isdecimal():
            continue
        try:
            fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
        except FileNotFoundError:
            continue
        if int(fields[2]) == group and fields[0] != 'Z':
            found.append(int(directory.name))
    return found


class NativeProcess:
    def __init__(self):
        # The SDK starts this object only inside its assigned GPU interpreter.
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if not visible.startswith('GPU-') or ',' in visible:
            raise RuntimeError('One SDK-assigned GPU UUID is required')
        self.environment = {key: value for key, value in os.environ.items() if not key.startswith('HUB_')}
        self.environment['AUDIO_GPU_UUID'] = visible
        self.groups = set()
        self.processes = {}

    def run(self, command, *, cwd=None, timeout=3600, log_path=None):
        if self.processes or any(members(g) for g in self.groups):
            raise RuntimeError('Native work is still unresolved')
        self.groups.clear()
        log = open(log_path, 'w') if log_path else None
        try:
            process = subprocess.Popen(command, cwd=cwd, env=self.environment,
                stdout=log or subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, start_new_session=True)
            self.groups.add(process.pid)
            self.processes[process.pid] = process
            try:
                output, _ = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                # No implicit kill/replay: completion marks the activity unknown,
                # retaining Hub ownership until the actual native process settles.
                raise TimeoutError('Native work has not reached its completion barrier') from None
            self.processes.pop(process.pid)
            if any(members(g) for g in self.groups):
                raise RuntimeError('Native descendants remain after the command exited')
            self.groups.clear()
            return {'returncode': process.returncode, 'stdout': output or ''}
        finally:
            if log:
                log.close()

    def complete(self):
        for pid, process in list(self.processes.items()):
            if process.poll() is not None:
                self.processes.pop(pid)
        if self.processes or any(members(g) for g in self.groups):
            raise RuntimeError('Native execution remains unresolved')
        self.groups.clear()


def load_model():
    return NativeProcess()


def completion(model):
    model.complete()


def cleanup():
    # Native model allocators belong to the child CLI and end on its real exit.
    gc.collect()
