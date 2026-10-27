"""CPU-only queue-context/restart fixtures; never execute a native model."""
import asyncio
import contextlib
import contextvars
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from hub_runtime import api
from ttd_model_runtime.protocol import HubError

marker=contextvars.ContextVar('fixture_request')
class RuntimeFixture:
    def __init__(self):self.completed=[]
    def complete_background_work(self):self.completed.append(marker.get(None))

class BackgroundOwnership(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory=tempfile.TemporaryDirectory();self.data=Path(self.directory.name)
        self.patches=[patch.object(api,'DATA',self.data),patch.object(api,'q',asyncio.Queue()),patch.object(api,'verification_key',lambda:'cpu-fixture-only')]
        for p in self.patches:p.start()
        self.runtime=RuntimeFixture();api.build_api(self.runtime)
        self.job='0'*32;api.write(self.data/self.job/'job.json',{'id':self.job,'status':'queued'})

    async def asyncTearDown(self):
        for p in reversed(self.patches):p.stop()
        self.directory.cleanup()

    async def consume(self,function):
        marker.set('request-a');await api.q.put((self.job,contextvars.copy_context()))
        self.assertEqual(self.runtime.pending_work(),1)
        with patch.object(api,'run_native',function):
            task=asyncio.create_task(api.worker())
            await asyncio.wait_for(api.q.join(),3)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):await task

    async def test_worker_and_completion_keep_the_original_request_context(self):
        def complete(jid):
            self.assertEqual(marker.get(),'request-a')
            state=api.read(jid);state['status']='completed';api.write(self.data/jid/'job.json',state)
        await self.consume(complete)
        self.assertEqual(self.runtime.completed,['request-a'])
        self.assertEqual(api.read(self.job)['status'],'completed')
        self.assertEqual(self.runtime.pending_work(),0)

    async def test_unknown_job_does_not_finish_background_ownership(self):
        def uncertain(jid):raise HubError('execution_unknown')
        await self.consume(uncertain)
        self.assertEqual(self.runtime.completed,[])
        self.assertEqual(api.read(self.job)['status'],'unknown')

    async def test_restart_does_not_replay_a_queued_native_job(self):
        state=api.read(self.job);state['_hub_actor_id']='cpu-fixture-actor';api.write(self.data/self.job/'job.json',state)
        with patch.dict(__import__('os').environ,{'HUB_RUNTIME_ACTOR_ID':'cpu-fixture-actor'}),patch.object(api.asyncio,'create_task',lambda coroutine:coroutine.close()):await api.boot()
        self.assertEqual(api.read(self.job)['status'],'interrupted')
        self.assertTrue(api.q.empty())

    async def test_candidate_startup_does_not_rewrite_legacy_running_jobs(self):
        state=api.read(self.job);state['status']='running';api.write(self.data/self.job/'job.json',state)
        with patch.dict(__import__('os').environ,{'HUB_RUNTIME_ACTOR_ID':'cpu-fixture-actor'}),patch.object(api.asyncio,'create_task',lambda coroutine:coroutine.close()):await api.boot()
        self.assertEqual(api.read(self.job)['status'],'running')
        self.assertTrue(api.q.empty())
