import pytest

from ml.evaluate import significance
from mockito import when, when2, unstub


class TestSignificanceInit:

    def test_guess_significance_test_from_experiment_dir(self, evaluation_model):
        assert significance.guess_significance_test_from_experiment_dir(evaluation_model(num_of_runs=4,
                                                                                         num_repetitions=1,
                                                                                         num_folds=1)) == [
                   'StuartMaxwell, McNemarBowker']
        assert significance.guess_significance_test_from_experiment_dir(evaluation_model(num_of_runs=4,
                                                                                         num_repetitions=2,
                                                                                         num_folds=1)) == []

        assert significance.guess_significance_test_from_experiment_dir(evaluation_model(num_of_runs=4,
                                                                                         num_repetitions=3,
                                                                                         num_folds=4)) == [
                   'Modified-t-Test with Bengio correctur']

    def test_guess_significance_test_from_experiment_dir_failure_cases(self, evaluation_model):
        ev_model_1 = evaluation_model(num_of_runs=4, num_repetitions=0, num_folds=1)
        when2(significance.determine_repetitions_and_folds, ev_model_1).thenReturn((0, 1))
        with pytest.raises(NotImplementedError):
            significance.guess_significance_test_from_experiment_dir(ev_model_1)
        unstub()

        ev_model_2 = evaluation_model(num_of_runs=4, num_repetitions=1, num_folds=-1)
        when(significance).determine_repetitions_and_folds(ev_model_2).thenReturn((1, -1))
        with pytest.raises(NotImplementedError):
            significance.guess_significance_test_from_experiment_dir(ev_model_2)
        unstub()

    def test_determine_repetitions_and_folds(self, evaluation_model):
        assert significance.determine_repetitions_and_folds(evaluation_model(num_of_runs=4,
                                                                             num_repetitions=1,
                                                                             num_folds=1)) == (1, 1)
        assert significance.determine_repetitions_and_folds(evaluation_model(num_of_runs=4,
                                                                             num_repetitions=5,
                                                                             num_folds=3)) == (5, 3)

    def test_determine_repetitions_and_folds_inconsistent_number_of_folds(self, evaluation_model):
        ev_model = evaluation_model(num_of_runs=4, num_repetitions=3, num_folds=2)
        ev_model.runs[0].repetitions[0].folds = []
        with pytest.raises(ValueError):
            significance.determine_repetitions_and_folds(ev_model)

    def test_determine_repetitions_and_folds_repetition_without_repetitions(self, evaluation_model):
        ev_model = evaluation_model(num_of_runs=4, num_repetitions=0, num_folds=4)
        with pytest.raises(AssertionError):
            significance.determine_repetitions_and_folds(ev_model)

    def test_determine_repetitions_and_folds_inconsistent_number_of_repetitions(self, evaluation_model):
        ev_model = evaluation_model(num_of_runs=4, num_repetitions=3, num_folds=4)
        del ev_model.runs[0].repetitions[2]
        with pytest.raises(ValueError):
            significance.determine_repetitions_and_folds(ev_model)

    def test_determine_repetitions_and_folds_no_runs(self, evaluation_model):
        ev_model = evaluation_model(num_of_runs=0, num_repetitions=2, num_folds=3)
        with pytest.raises(AssertionError):
            significance.determine_repetitions_and_folds(ev_model)

    def test_interpret_p_value_variant_2(self):
        assert significance.interpret_p_value_variant_2(0.02) == 'Null hypothesis rejected. There is significant ' \
                                                                 'difference.'
        assert significance.interpret_p_value_variant_2(0.05) == 'Null hypothesis rejected. There is significant ' \
                                                                 'difference.'
        assert significance.interpret_p_value_variant_2(1) == 'Null hypothesis can not be rejected. There is not ' \
                                                              'significant difference!'

    def test_interpret_p_value_variant_1(self):
        assert significance.interpret_p_value_variant_1(0.8) == 'Weak or no evidence'
        assert significance.interpret_p_value_variant_1(0.10) == 'Moderate evidence'
        assert significance.interpret_p_value_variant_1(0.08) == 'Moderate evidence'
        assert significance.interpret_p_value_variant_1(0.05) == 'Strong evidence'
        assert significance.interpret_p_value_variant_1(0.02) == 'Strong evidence'
        assert significance.interpret_p_value_variant_1(0.01) == 'Very strong evidence'
        assert significance.interpret_p_value_variant_1(0.00032) == 'Very strong evidence'
