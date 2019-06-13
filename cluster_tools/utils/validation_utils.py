import json


class SegmentationValidation:

    def load_rand_primitives(self, eval_dict):
        assert len({'randA', 'randB', 'randAB'} - set(eval_dict.keys())) == 0, "Not all rand primitives present"
        self.rand_a = eval_dict['randA']
        self.rand_b = eval_dict['randB']
        self.rand_ab = eval_dict['randAB']

    def load_vi_primitives(self, eval_dict):
        assert len({'viA', 'viB', 'viAB'} - set(eval_dict.keys())) == 0, "Not all vi primitives present"
        self.vi_a = eval_dict['viA']
        self.vi_b = eval_dict['viB']
        self.vi_ab = eval_dict['viAB']

    def __init__(self, path):
        with open(path, 'r') as f:
            eval_dict = json.load(f)
        self.load_rand_primitives(eval_dict)
        self.load_vi_primitives(eval_dict)
        assert 'nPoints' in eval_dict, "Number of points not present"
        self.n_points = eval_dict['nPoints']

    #
    # rand measures
    #

    @property
    def rand_index(self):
        return 1. - (self.rand_a + self.rand_b - 2 * self.rand_ab) / (self.n_points * self.n_points)

    @property
    def rand_precision(self):
        return self.rand_ab / self.rand_b

    @property
    def rand_recall(self):
        return self.rand_ab / self.rand_a

    @property
    def adapated_rand_score(self):
        prec = self.rand_precision
        rec = self.rand_recall
        return 2. * prec * rec / (prec + rec)

    #
    # vi measures
    #

    @property
    def variation_of_information(self):
        return self.vi_a + self.vi_b - 2. * self.vi_a

    # TODO
    @property
    def vi_merge(self):
        return

    @property
    def vi_split(self):
        return
