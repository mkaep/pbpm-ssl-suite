from ml.pipeline import job_executor
from ml.persistence import json
import pytest
import os
import subprocess
from mockito import when, mock


class TestJobExecutor:

    def test_init_with_jobs(self, job, tmp_path):
        jobs_directory = tmp_path / 'job_directory'
        job_1 = job(job_directory='job_1')
        job_2 = job(job_directory='job_2')
        jobs = [job_1, job_2]

        executor = job_executor.JobExecutor(str(jobs_directory), jobs)
        assert executor._jobs_directory == str(jobs_directory)
        assert executor._verbose is False
        assert executor._jobs_done == []
        assert executor._jobs_failed == []
        assert executor._jobs_waiting == jobs

    def test_init_with_no_jobs(self, job, tmp_path):
        jobs_directory = str(tmp_path / 'job_directory')

        when(os).listdir(jobs_directory).thenReturn(['jobs.jsonl', 'jobs_failed.jsonl', 'jobs_done.jsonl',
                                                     'jobs_waiting.jsonl'])
        # Mock the JSON importers
        jobs_done = [job() for _ in range(3)]
        jobs_failed = [job() for _ in range(1)]
        jobs_waiting = [job() for _ in range(2)]

        mock_importer_done = mock()
        when(json).JsonJobImporter(os.path.join(jobs_directory, 'jobs_done.jsonl')).thenReturn(mock_importer_done)
        when(mock_importer_done).load().thenReturn(jobs_done)

        mock_importer_failed = mock()
        when(json).JsonJobImporter(os.path.join(jobs_directory, 'jobs_failed.jsonl')).thenReturn(mock_importer_failed)
        when(mock_importer_failed).load().thenReturn(jobs_failed)

        mock_importer_waiting = mock()
        when(json).JsonJobImporter(os.path.join(jobs_directory, 'jobs_waiting.jsonl')).thenReturn(mock_importer_waiting)
        when(mock_importer_waiting).load().thenReturn(jobs_waiting)

        executor = job_executor.JobExecutor(jobs_directory)
        assert executor._jobs_directory == jobs_directory
        assert executor._verbose is False
        assert executor._jobs_done == jobs_done
        assert executor._jobs_failed == jobs_failed
        assert executor._jobs_waiting == jobs_waiting

    def test_init_failure_case_with_no_jobs(self, tmp_path):
        jobs_directory = tmp_path / 'job_directory'

        when(os).listdir(str(jobs_directory)).thenReturn([])
        # reason: files not found in directory
        with pytest.raises(AssertionError):
            job_executor.JobExecutor(str(jobs_directory))

    def test_run(self, job, tmp_path):
        jobs_directory = tmp_path / 'job_directory'
        job_1 = job(job_directory='job_1')
        job_2 = job(job_directory='job_2')
        jobs = [job_1, job_2]

        executor = job_executor.JobExecutor(str(jobs_directory), jobs)
        assert executor._jobs_done == []
        assert executor._jobs_failed == []
        assert executor._jobs_waiting == jobs

        when(executor).run_job(job_1).thenReturn(True)
        when(executor).run_job(job_2).thenReturn(False)

        executor.run()

        assert executor._jobs_done == [job_1]
        assert executor._jobs_failed == [job_2]
        assert executor._jobs_waiting == []
        assert os.path.isfile(os.path.join(jobs_directory, 'jobs_done.jsonl'))
        assert os.path.isfile(os.path.join(jobs_directory, 'jobs_failed.jsonl'))
        assert os.path.isfile(os.path.join(jobs_directory, 'jobs_waiting.jsonl'))

    def test_run_complete_successfully(self, job, tmp_path):
        jobs_directory = tmp_path / 'job_directory'
        job_1 = job(job_directory='job_1')
        job_2 = job(job_directory='job_2')
        jobs = [job_1, job_2]

        executor = job_executor.JobExecutor(str(jobs_directory), jobs)
        assert executor._jobs_done == []
        assert executor._jobs_failed == []
        assert executor._jobs_waiting == jobs

        when(executor).run_job(job_1).thenReturn(True)
        when(executor).run_job(job_2).thenReturn(True)

        executor.run()

        assert executor._jobs_done == [job_1, job_2]
        assert executor._jobs_failed == []
        assert executor._jobs_waiting == []
        assert os.path.isfile(os.path.join(jobs_directory, 'jobs_done.jsonl'))
        assert os.path.isfile(os.path.join(jobs_directory, 'jobs_failed.jsonl'))
        assert os.path.isfile(os.path.join(jobs_directory, 'jobs_waiting.jsonl'))

    def test_retry_failed_executions(self, job, tmp_path):
        jobs_directory = tmp_path / 'job_directory'
        job_1 = job(job_directory='job_1')
        job_2 = job(job_directory='job_2')
        jobs_failed = [job_1, job_2]

        executor = job_executor.JobExecutor(str(jobs_directory), [])
        executor._jobs_failed = jobs_failed
        executor._jobs_done = []
        executor._jobs_waiting = []

        when(executor).run_job(job_1).thenReturn(True)
        when(executor).run_job(job_2).thenReturn(False)

        executor.retry_failed_executions()

        assert executor._jobs_done == [job_1]
        assert executor._jobs_failed == [job_2]
        assert executor._jobs_waiting == []
        assert os.path.isfile(os.path.join(jobs_directory, 'jobs_done.jsonl'))
        assert os.path.isfile(os.path.join(jobs_directory, 'jobs_failed.jsonl'))
        assert os.path.isfile(os.path.join(jobs_directory, 'jobs_waiting.jsonl'))

    def test_reload_execution_state(self, job):
        jobs_directory = 'res/jobs_directory'
        when(os).listdir(jobs_directory).thenReturn(['jobs.jsonl', 'jobs_failed.jsonl', 'jobs_done.jsonl',
                                                     'jobs_waiting.jsonl'])
        # Mock the JSON importers
        jobs_done = [job() for _ in range(3)]
        jobs_failed = [job() for _ in range(1)]
        jobs_waiting = [job() for _ in range(2)]

        mock_importer_done = mock()
        when(json).JsonJobImporter(os.path.join(jobs_directory, 'jobs_done.jsonl')).thenReturn(mock_importer_done)
        when(mock_importer_done).load().thenReturn(jobs_done)

        mock_importer_failed = mock()
        when(json).JsonJobImporter(os.path.join(jobs_directory, 'jobs_failed.jsonl')).thenReturn(mock_importer_failed)
        when(mock_importer_failed).load().thenReturn(jobs_failed)

        mock_importer_waiting = mock()
        when(json).JsonJobImporter(os.path.join(jobs_directory, 'jobs_waiting.jsonl')).thenReturn(mock_importer_waiting)
        when(mock_importer_waiting).load().thenReturn(jobs_waiting)

        executor = job_executor.JobExecutor(jobs_directory, [])
        executor.reload_execution_state()

        assert executor._jobs_done == jobs_done
        assert executor._jobs_failed == jobs_failed
        assert executor._jobs_waiting == jobs_waiting

    def test_failure_cases_reload_execution_state(self):
        corrupt_jobs_directory = 'res/corrupt_jobs_directory'
        when(os).listdir(corrupt_jobs_directory).thenReturn(['jobs.jsonl', 'jobs_failed.jsonl', 'jobs_done.jsonl'])
        executor = job_executor.JobExecutor(corrupt_jobs_directory, [])
        # reason: missing jobs_waiting_file
        with pytest.raises(AssertionError):
            executor.reload_execution_state()

        when(os).listdir(corrupt_jobs_directory).thenReturn(['jobs.jsonl', 'jobs_waiting.jsonl', 'jobs_done.jsonl'])
        # reason: missing jobs_failed_file
        with pytest.raises(AssertionError):
            executor.reload_execution_state()

        when(os).listdir(corrupt_jobs_directory).thenReturn(['jobs_failed.jsonl', 'jobs_waiting.jsonl',
                                                             'jobs_done.jsonl'])
        # reason: missing jobs_file
        with pytest.raises(AssertionError):
            executor.reload_execution_state()

        when(os).listdir(corrupt_jobs_directory).thenReturn(['jobs_failed.jsonl', 'jobs_waiting.jsonl',
                                                             'jobs.jsonl'])
        # reason: missing jobs_done_file
        with pytest.raises(AssertionError):
            executor.reload_execution_state()

    def test_save_execution_state(self, tmp_path, job):
        jobs_directory = str(tmp_path / 'job_directory')

        executor = job_executor.JobExecutor(str(jobs_directory), [])
        executor._jobs_failed = [job() for _ in range(1)]
        executor._jobs_done = [job() for _ in range(3)]
        executor._jobs_waiting = [job() for _ in range(2)]

        executor.save_execution_state()

        assert os.path.isfile(os.path.join(jobs_directory, 'jobs_done.jsonl'))
        assert os.path.isfile(os.path.join(jobs_directory, 'jobs_failed.jsonl'))
        assert os.path.isfile(os.path.join(jobs_directory, 'jobs_waiting.jsonl'))

    def test_run_job(self, tmp_path, job):
        jobs_directory = tmp_path / 'job_directory'
        job_to_execute = job(job_directory='test_job')

        path = os.path.join(jobs_directory, '.tmp', 'run_job.bat')
        when(subprocess).run(str(path), shell=True,
                             check=True).thenReturn(True)

        executor = job_executor.JobExecutor(str(jobs_directory), [])
        status = executor.run_job(job_to_execute)

        assert os.path.isdir(os.path.join(jobs_directory, '.tmp'))
        assert os.path.isfile(os.path.join(jobs_directory, '.tmp', 'run_job.bat'))
        assert status is True

    def test_run_job_failure_case(self, tmp_path, job):
        jobs_directory = tmp_path / 'job_directory'
        job_to_execute = job(job_directory='test_job')

        path = os.path.join(jobs_directory, '.tmp', 'run_job.bat')
        when(subprocess).run(str(path), shell=True,
                             check=True).thenRaise(subprocess.CalledProcessError(cmd='failed_command', returncode=-1))

        executor = job_executor.JobExecutor(str(jobs_directory), [])
        status = executor.run_job(job_to_execute)

        assert os.path.isdir(os.path.join(jobs_directory, '.tmp'))
        assert os.path.isfile(os.path.join(jobs_directory, '.tmp', 'run_job.bat'))
        assert status is False
