bms_parser = require('./parser-full');

if (process.argv.length < 3) {
    console.log('[Fail]BMS parser Usage: node ' + process.argv[1] + ' FILENAME');
    process.exit(1);
}
console.log('[Log]BMS parser version 0.1.0');
// Read the file and print its contents.
var fs = require('fs')
    , filename = process.argv[2];

let parser = new bms_parser();

parser.loadfile(filename);
parser.parse();
console.log(parser.timednotes);