# Config.py
# a config manager interface built using Kaptan.

from kaptan import Kaptan


class Config:
    def __init__(self):
        import os
        script_path = os.path.dirname(os.path.abspath(__file__))
        self.path_to_config = script_path + "/config.json"
        self.config = Kaptan(handler="json")

    # def write_config(self):
    #     f = open(self.path_to_config,'w')
    #     f.write(self.config.export("json"))
    #     f.close()

    def read_config(self):
        f = open(self.path_to_config, 'r')
        self.config.import_config(f.read())
        f.close()

    def __getitem__(self, item):
        self.read_config()
        return self.config.get(item)

    # overrides the . operator.
    def __getattr__(self, item):
        return self.__getitem__(item)


config = Config()
