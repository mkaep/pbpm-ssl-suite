import os
import shutil
from ml import main


# class TestMain:
#
#     def test_create_basic_folder_structure_1(self, experiment, dataset, approach):
#         exp = experiment(data_dir='../tests/main/res/data', run_dir='../tests/main/res/run',
#                          evaluation_dir='../tests/main/res/evaluation',
#                          event_logs=[
#                              dataset(name='Dataset_1'),
#                              dataset(name='Dataset_2')
#                          ],
#                          approaches=[
#                              approach(name='Approach_1'),
#                              approach(name='Approach_2')
#                          ])
#
#         main._create_basic_folder_structure(exp)
#
#         # Check the directory structure whether as expected
#         run_dir = '../tests/main/res/run'
#         assert os.path.exists(run_dir)
#         assert os.path.exists(os.path.join(run_dir, 'test_experiment'))
#
#         assert len(os.listdir(os.path.join(run_dir, 'test_experiment'))) == 2
#         assert os.path.exists(os.path.join(run_dir, 'test_experiment', 'Dataset_1'))
#         assert os.path.exists(os.path.join(run_dir, 'test_experiment', 'Dataset_2'))
#
#         assert len(os.listdir(os.path.join(run_dir, 'test_experiment', 'Dataset_1'))) == 3
#         assert os.path.exists(os.path.join(run_dir, 'test_experiment', 'Dataset_1', '_common_data'))
#         assert os.path.exists(os.path.join(run_dir, 'test_experiment', 'Dataset_1', 'Approach_1'))
#         assert os.path.exists(os.path.join(run_dir, 'test_experiment', 'Dataset_1', 'Approach_2'))
#
#         assert len(os.listdir(os.path.join(run_dir, 'test_experiment', 'Dataset_2'))) == 3
#         assert os.path.exists(os.path.join(run_dir, 'test_experiment', 'Dataset_2', '_common_data'))
#         assert os.path.exists(os.path.join(run_dir, 'test_experiment', 'Dataset_2', 'Approach_1'))
#         assert os.path.exists(os.path.join(run_dir, 'test_experiment', 'Dataset_2', 'Approach_2'))
#
#         # After successful test remove created folders
#         shutil.rmtree('../tests/main/res/run')
#
#     def test_create_basic_folder_structure_2(self, experiment, dataset, approach):
#         exp = experiment(data_dir='../tests/main/res/data', run_dir='../tests/main/res/run_incomplete',
#                          evaluation_dir='../tests/main/res/evaluation',
#                          event_logs=[
#                              dataset(name='Dataset_1'),
#                              dataset(name='Dataset_2')
#                          ],
#                          approaches=[
#                              approach(name='Approach_1'),
#                              approach(name='Approach_2')
#                          ])
#
#         main._create_basic_folder_structure(exp)
#
#         # Check the directory structure whether as expected
#         run_dir = '../tests/main/res/run_incomplete'
#         assert os.path.exists(run_dir)
#         assert os.path.exists(os.path.join(run_dir, 'test_experiment'))
#
#         assert len(os.listdir(os.path.join(run_dir, 'test_experiment'))) == 2
#         assert os.path.exists(os.path.join(run_dir, 'test_experiment', 'Dataset_1'))
#         assert os.path.exists(os.path.join(run_dir, 'test_experiment', 'Dataset_2'))
#
#         assert len(os.listdir(os.path.join(run_dir, 'test_experiment', 'Dataset_1'))) == 3
#         assert os.path.exists(os.path.join(run_dir, 'test_experiment', 'Dataset_1', '_common_data'))
#         assert os.path.exists(os.path.join(run_dir, 'test_experiment', 'Dataset_1', 'Approach_1'))
#         assert os.path.exists(os.path.join(run_dir, 'test_experiment', 'Dataset_1', 'Approach_2'))
#
#         assert len(os.listdir(os.path.join(run_dir, 'test_experiment', 'Dataset_2'))) == 3
#         assert os.path.exists(os.path.join(run_dir, 'test_experiment', 'Dataset_2', '_common_data'))
#         assert os.path.exists(os.path.join(run_dir, 'test_experiment', 'Dataset_2', 'Approach_1'))
#         assert os.path.exists(os.path.join(run_dir, 'test_experiment', 'Dataset_2', 'Approach_2'))
#
#         # After successful test remove created folders
#         os.rmdir(os.path.join(run_dir, 'test_experiment', 'Dataset_1', '_common_data'))
#         os.rmdir(os.path.join(run_dir, 'test_experiment', 'Dataset_2', '_common_data'))
#         os.rmdir(os.path.join(run_dir, 'test_experiment', 'Dataset_2', 'Approach_2'))






