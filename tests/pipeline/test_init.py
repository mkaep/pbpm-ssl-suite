import os
from ml import pipeline


class TestInit:

    def test_create_basic_folder_structure_without_datasets(self, tmp_path, experiment):
        pipeline.create_basic_folder_structure(experiment(name='Test-Experiment',
                                                          run_dir=str(tmp_path),
                                                          event_logs=[]))

        assert os.path.isdir(os.path.join(tmp_path, 'Test-Experiment')) is True
        assert os.listdir(os.path.join(tmp_path, 'Test-Experiment')) == []

    def test_create_basic_folder_structure_with_datasets(self, tmp_path, experiment, dataset, approach):
        pipeline.create_basic_folder_structure(experiment(name='Test-Experiment',
                                                          run_dir=str(tmp_path),
                                                          event_logs=[
                                                              dataset(name='Test_Log_1'),
                                                              dataset(name='Test_Log_2')
                                                          ],
                                                          approaches=[
                                                              approach(name='Approach_1'),
                                                              approach(name='Approach_2'),
                                                              approach(name='Approach_3')
                                                          ]))
        assert os.path.isdir(os.path.join(tmp_path, 'Test-Experiment')) is True
        assert set(os.listdir(os.path.join(tmp_path, 'Test-Experiment'))) == {'Test_Log_2', 'Test_Log_1'}
        assert set(os.listdir(os.path.join(tmp_path, 'Test-Experiment', 'Test_Log_1'))) == {'_common_data',
                                                                                            'Approach_1',
                                                                                            'Approach_2',
                                                                                            'Approach_3'}
        assert set(os.listdir(os.path.join(tmp_path, 'Test-Experiment', 'Test_Log_2'))) == {'_common_data',
                                                                                            'Approach_1',
                                                                                            'Approach_2',
                                                                                            'Approach_3'}
        # Check if the directory are empty
        assert os.listdir(os.path.join(tmp_path, 'Test-Experiment', 'Test_Log_1', '_common_data')) == []
        assert os.listdir(os.path.join(tmp_path, 'Test-Experiment', 'Test_Log_1', 'Approach_1')) == []
        assert os.listdir(os.path.join(tmp_path, 'Test-Experiment', 'Test_Log_1', 'Approach_2')) == []
        assert os.listdir(os.path.join(tmp_path, 'Test-Experiment', 'Test_Log_1', 'Approach_3')) == []

        assert os.listdir(os.path.join(tmp_path, 'Test-Experiment', 'Test_Log_2', '_common_data')) == []
        assert os.listdir(os.path.join(tmp_path, 'Test-Experiment', 'Test_Log_2', 'Approach_1')) == []
        assert os.listdir(os.path.join(tmp_path, 'Test-Experiment', 'Test_Log_2', 'Approach_2')) == []
        assert os.listdir(os.path.join(tmp_path, 'Test-Experiment', 'Test_Log_2', 'Approach_3')) == []



