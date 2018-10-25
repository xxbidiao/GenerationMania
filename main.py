from pick import pick
from scripts import MongoDBInterface, DataImportMain, Config, createChart
import os


OPTION_BACK = "Back"

def get_first_level(options):
    result = []
    if type(options) is str:
        #This is the command itself.
        return options
    for key in options:
        result.append(key)
    return sorted(result)

def find_menu(options,selected):
    result = None
    if selected == []:
        result = get_first_level(options)
    else:
        opt = options
        for item in selected:

            opt = opt[item]
        result = get_first_level(opt)
        if type(result) is list:
            result.append(OPTION_BACK)
    return result

def combine_selected(selected):
    str = "root"
    for item in selected:
        str += "->"
        str += item
    return str

def get_choice(options,selected):
    title = "Current selection: "+combine_selected(selected)
    true_options = find_menu(options,selected)
    if type(true_options) is str:
        return selected
    option,index = pick(true_options,title,indicator="->")
    result = selected
    result.append(option)
    return result

options_list = {
    "1.Read Instructions":{
        "1.Introduction":"instruction",
        "2.Dependencies":"dependencies"
    },
    "2.Setup data":{
        "1.Download data":"download_data",
        "2.Instructions on installing MongoDB":"install_db",
        "3.Test DB connection":"test_db",
        "4.Importing data into database":"import_db"
    },
    "3.Sample Classification":{
        "1.Do Audio Classification":"do_audio",
        "2.Do Pitch Classification(unused)":"do_pitch"
    },
    "4.Playable Classification":{
        "1.Generate dataset for playable classification":"gen_pc_data",
        "2.Train using our feedforward+summary model":"train_ff",
        "3.Train using our LSTM model":"train_lstm",
        "4.Delete trained models":"del_states"
    },
    "5.Evaluation":{
        "1.Evaluate per-chart accuracy on Playable Classification(Feed forward)":"eval_ff",
        "2.Evaluate per-chart accuracy on Playable Classification(LSTM)": "eval_lstm",
        "3.Get reconstructed chart":"reconstruct"
    },
    "6.Generation (Experimental)":{
        "1.Train step generation model":"train_cols",
        "2.Evaluate step generation model":"eval_cols",
        "3.Mix two charts":"mix"
    },
    "7.Exit":"exit"
}

def main():
    selected = []
    while True:
        selected = get_choice(options_list,selected)
        if selected[-1] == OPTION_BACK:
            selected = selected[:-2]
        if type(find_menu(options_list,selected)) is str:
            # print("Final selection:"+str(combine_selected(selected)))
            # exit()
            call_submodules(find_menu(options_list,selected))
            input("Press Enter to continue...")
            selected = []

def call_submodules(command):
    #clear the screen before executing a sub command
    print("\033[H\033[J")

    if command == "exit":
        exit()
    elif command == "instruction":
        print("This is the entry point for GenerationMania, a chart generator.")
    elif command == "download_data":
        print("Create a new folder named 'rawdata' in the same directory of this file. In this folder create a new folder named `bof2011`.")
        print("Download data using the torrent provided in the root folder or use this magnet link, to rawdata/bof2011:")
        print("magnet:?xt=urn:btih:d133a79e03ff1c11c9512739542fe25a1cd2f03d&dn=%5BBMS%5D%5BPACK%5D%20THE%20BMS%20OF%20FIGHTERS%202011%20-%20Intersection%20of%20conflict%20-&tr=http%3A%2F%2Fwww.ceena.net%2Fannounce.php")
        print("After download is finished, extract every zip file so that they have their individual folders.")
    elif command == "dependencies":
        f = open("DEPENDENCIES", "r")
        print(f.read())
    elif command == "install_db":
        print("Visit https://docs.mongodb.com/manual/installation/ for how to get MongoDB.")
    elif command == "test_db":
        MongoDBInterface.test_database()
    elif command == "import_db":
        DataImportMain.import_data("rawdata/")
    elif command == "do_audio":
        print("Wait for a while, importing tensorflow...")
        from scripts import AudioInterface
        AudioInterface.do_audio()
    elif command == "do_pitch":
        print("Wait for a while, importing tensorflow...")
        from scripts import AudioInterface
        AudioInterface.do_audio(type="pitch")
    elif command == "gen_pc_data":
        print("Wait for a while, importing pytorch...")
        from scripts import DBtoLearn as DL
        outfile = Config.config.training_file
        outfile_pc = Config.config.training_file_per_chart
        DL.preparePlayableWithLookbackFile(
            dict(outfile=outfile, outfile_per_chart=outfile_pc, extra_info='yeah', per_note_diff_osu='yeah')
            , algorithm="osu")
    elif command == "train_ff":
        print("Wait for a while, importing pytorch...")
        from scripts import model_feedforward as ff
        ff.train_model()
    elif command == "eval_ff":
        print("Wait for a while, importing pytorch...")
        from scripts import generatePerChartResult as gp
        gp.eval_all()
    elif command == "train_lstm":
        print("Wait for a while, importing pytorch...")
        from scripts import model_lstm as lstm
        lstm.train()
    elif command == "eval_lstm":
        print("Wait for a while, importing pytorch...")
        from scripts import model_lstm as lstm
        lstm.eval()
    elif command == "del_states":
        print("Deleting states...")
        try:
            os.remove(Config.config.training_state_feedforward)
        except FileNotFoundError:
            print("State for feedforward network missing.")
        try:
            os.remove(Config.config.training_state_LSTM)
        except FileNotFoundError:
            print("State for LSTM network missing.")
        # from scripts import model_lstm as lstm
        # lstm.train()
        print("Done.")
    elif command == "train_cols":
        print("Wait for a while, importing pytorch...")
        from scripts import model_ff_cols as ffc
        ffc.train_model()
    elif command == "eval_cols":
        print("Wait for a while, importing pytorch...")
        from scripts import generatePerChartResult_column as gc
        gc.eval_all()
    elif command == "mix":
        print("Wait for a while, importing pytorch & tensorflow...")
        from scripts import createChart as cc
        aud = input("Mix Base: Enter path to a BMS where its audio content would be used: ")
        sty = input("Mix With: Enter path to a BMS where its style would be used: ")
        result = cc.mix_two_charts(audio_provider=aud,style_provider=sty)
        cc.create_bmson_package(bmson=result,audio_source=aud)


    else:
        print("%s:Coming soon!"%command)


if __name__ == "__main__":
    main()
