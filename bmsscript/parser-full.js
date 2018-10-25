"use strict"
var bms = require('./bms-js/');
var fs = require('fs')
var iconv = require('iconv-lite')

class bms_parser {
    constructor()
    {

    }

    loadfile(filename)
    {
        this.bmsdata = fs.readFileSync(filename)
        //this.bmsdata = iconv.decode(fs.readFileSync(filename),"SHIFT_JIS");
        //this.bmsdata = unescape(encodeURIComponent(this.bmsdata));

    }

    parse()
    {
        this.bmstext = bms.Reader.read(this.bmsdata);
        var compile_result = bms.Compiler.compile(this.bmstext);
        this.chart = compile_result.chart;
        this.timing = bms.Timing.fromBMSChart(this.chart);
        this.notes = bms.Notes.fromBMSChart(this.chart);
        this.parse_warnings = compile_result.warnings;
        this.timednotes = []
        for(var note in this.notes._notes)
        {
            var time_of_note = this.timing.beatToSeconds(this.notes._notes[note].beat);
            var end_of_note = this.timing.beatToSeconds(this.notes._notes[note].endBeat);
            this.timednotes.push({time:time_of_note,end:end_of_note,data:this.notes._notes[note]});
        }
    }
}

module.exports = bms_parser;


