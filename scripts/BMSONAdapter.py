# data structures for bmson
# Functions in this file helps generating a .bmson file.
# It may not be supported by all BMS players, though.
# Suggested player: BEMUSE
# Get it from https://bemuse.ninja/

SECONDS_IN_A_MINUTE = 60

# BMSON representation.
class bmson:
    def __init__(self):
        self.doc = None
        self.init_default()

    # setup default bmson file.
    def init_default(self):
        self.doc = \
        {
            "bga": {
                "bga_events": [],
                "bga_header": [],
                "layer_events": [],
                "poor_events": []
            },
            "bpm_events": [], #{y,bpm}
            "info": {
                "artist": "GenerationMania",
                "back_image": "",
                "banner_image": "",
                "chart_name": "",
                "eyecatch_image": "",
                "genre": "",
                "init_bpm": 0,
                "judge_rank": 100,
                "level": 3,
                "mode_hint": "beat-7k",
                "resolution": 240,
                "subartists": [],
                "subtitle": "",
                "title": "",
                "total": 100
            },
            "lines":None, #intentionally set to None for default behaviour #{y}
            "sound_channels":[], #{name,[{x(COL,0 as unplayable,y,l,c(False for BMS)}]}
            "stop_events":[], #{y,duration}
            "version":"1.0.0"
        }

    def get_resolution(self):
        return self.doc["info"]["resolution"]

    # Add a line for each measure.
    def add_default_lines(self):
        pass # by default, bmson players are required to add default lines if lines is not provided.

    # get/set the initial bpm of the stage.
    def set_bpm(self,val):
        self.doc["info"]["init_bpm"] = val

    def get_bpm(self):
        return self.doc["info"]["init_bpm"]

    # assuming the song is always at initial BPM, which beat signature should this note be at?
    def get_invariable_pulse(self,time):
        return time * self.get_resolution() * self.get_bpm() / SECONDS_IN_A_MINUTE

    # add note using GenerationMania intermediate format.
    def add_note(self,note,column = None):
        sound_file = note["sound_file"]
        if column is None:
            x = note["gen_column"]
        else:
            x = column
        y = int(self.get_invariable_pulse(note["time"]))
        c = False
        l = 0
        found = False
        for element in self.doc["sound_channels"]:
            if element['name'] == sound_file:
                found = True
                element['notes'].append(
                    {
                        "x":x,
                        "y":y,
                        "l":l,
                        "c":c
                    }
                )
        if not found:
            self.doc['sound_channels'].append(
                {
                    "name":sound_file,
                    "notes":[
                        {
                            "x": x,
                            "y": y,
                            "l": l,
                            "c": c
                        }
                    ]
                }
            )

# It is not possible to convert internal bms back to BMS easily... But this generates a .bmson file.
# generate a bmson file using internal bms representation
# presented in document file.
# data: The body of the bms.
# header: The header of the bms.
def generate_bmson(data,header):
    result = bmson()
    print("Raw Header:\n%s"%header)
    result.doc['info']['title'] = header['title']
    result.doc['info']['genre'] = header['genre']
    result.doc['info']['level'] = 98
    result.set_bpm(float(header["bpm"]))
    for datum in data:
        if datum['gen_column'] < 0:
            result.add_note(datum,0)
        else:
            result.add_note(datum,datum['gen_column'])
    return result
