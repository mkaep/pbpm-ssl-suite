import os

import pytest
from ml.core import loader
from pm4py.objects.log.obj import EventLog


# TODO Add tests for .gz archives
class TestLoader:

    def test_can_import_from(self):
        file_path_1 = ''
        assert loader.Loader.can_import_from(file_path_1) is False

        file_path_2 = 'ggf:/gfgfg'
        assert loader.Loader.can_import_from(file_path_2) is False

        file_path_3 = 'tests/core/res/sample_log_1.xes'
        assert loader.Loader.can_import_from(file_path_3) is True

        file_path_4 = 'tests/core/res/test.csv'
        assert loader.Loader.can_import_from(file_path_4) is False

    def test_failure_load_from_zip_archive_no_xes_files(self):
        # reason: archive does not contain a .xes file
        with pytest.raises(ValueError):
            file_path = 'tests/core/res/archives/test_zip.zip'
            loader.Loader._load_from_zip_archive(file_path)

    def test_failure_load_from_zip_archive_more_than_one_file(self):
        # reason: More than one file'
        with pytest.raises(ValueError):
            file_path = 'tests/core/res/archives/test_zip_mixed.zip'
            loader.Loader._load_from_zip_archive(file_path)

    def test_load_from_zip_archive_3(self):
        file_path = 'tests/core/res/archives/sample_log.zip'
        event_log = loader.Loader._load_from_zip_archive(file_path)
        assert isinstance(event_log, EventLog)
        os.remove('tests/core/res/archives/sample_log.xes')

    def test_failure_load_from_zip_archive_no_xes_file(self):
        # reason: archive does not contain a .xes file
        with pytest.raises(ValueError):
            file_path = 'tests/core/res/archives/test_single_wrong.zip'
            loader.Loader._load_from_zip_archive(file_path)

    def test_load_event_log_1(self):
        file_path_1 = 'tests/core/res/sample_log_1.xes'
        event_log_1 = loader.Loader.load_event_log(file_path_1)
        assert isinstance(event_log_1, EventLog)

        file_path_2 = 'tests/core/res/archives/sample_log.zip'
        event_log_2 = loader.Loader._load_from_zip_archive(file_path_2)
        assert isinstance(event_log_2, EventLog)

    def test_failure_load_event_log(self):
        # reason: not supported file extension
        with pytest.raises(NotImplementedError):
            file_path = 'tests/core/res/sample_log_1.csv'
            loader.Loader.load_event_log(file_path)
