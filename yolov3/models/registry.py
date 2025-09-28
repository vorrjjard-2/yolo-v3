class Registry:
    def __init__(self, name: str):
        self._name = name
        self._module_dict = {}

    def register(self, name: str = None):
        def _register(obj):
            key = name or obj.__name__
            if key in self._module_dict:
                raise KeyError(f"{key} already registered in {self._name}")
            self._module_dict[key] = obj
            return obj
        return _register

    def __getitem__(self, key: str):
        if key not in self._module_dict:
            raise KeyError(f"{key} not found in {self._name}. "
                           f"Available: {list(self._module_dict.keys())}")
        return self._module_dict[key]

    def __repr__(self):
        return f"Registry({self._name}, items={list(self._module_dict.keys())})"

MODEL     = Registry("MODEL")
BACKBONE  = Registry("BACKBONE")
NECK      = Registry("NECK")
HEAD      = Registry("HEAD")
LOSS      = Registry("LOSS")
DATASET   = Registry("DATASET")
OPTIMIZER = Registry("OPTIMIZER")
SCHEDULER = Registry("SCHEDULER")

