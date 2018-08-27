import luigi


class DummyTarget(luigi.Target):
    """ Dummy target that always exists
    """
    def exists(self):
        return True


class DummyTask(luigi.Task):
    """ Dummy Task for dependencies that are always true
    """
    def output(self):
        return DummyTarget()
