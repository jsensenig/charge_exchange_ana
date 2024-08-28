class CutFactory:
    def __init__(self):
        self._builders = {}

    def register_builder(self, key, builder):
        self._builders[key] = builder

    def create(self, key, config, cut_name):
        builder = self._builders.get(key)
        if not builder:
            raise ValueError(key)
        return builder(config, cut_name)

