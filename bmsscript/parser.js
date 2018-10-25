"use strict";
var tools = require('./bms');
var fs = require('fs')

class bms_parser {
    constructor()
    {

    }

    loadfile(filename)
    {
        this.bmsdata = fs.readFileSync(filename, 'utf8');
    }

    parse()
    {
        this.parsedResult = tools.parse(this.bmsdata);
        this.parsedheaders = this.parsedResult.headers;
        this.allEvents = this.parsedResult.events;
    }

}


if (process.argv.length < 3) {
    console.log('[Fail]BMS parser Usage: node ' + process.argv[1] + ' FILENAME');
    process.exit(1);
}
//console.log('[Log]BMS parser version 0.1.0');
// Read the file and print its contents.
var fs = require('fs')
    , filename = process.argv[2];

let parser = new bms_parser();

parser.loadfile(filename);
parser.parse();
var events = JSON.stringify(parser);
console.log(events);


