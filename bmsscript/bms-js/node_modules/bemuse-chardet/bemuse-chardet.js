
var utf8  = require('./encoding/utf8'),
  unicode = require('./encoding/unicode'),
  mbcs    = require('./encoding/mbcs'),
  sbcs    = require('./encoding/sbcs'),
  iso2022 = require('./encoding/iso2022');

var self = this;

var recognisers = [
  new utf8,
  new unicode.UTF_16BE,
  new unicode.UTF_16LE,
  new mbcs.sjis,
  new mbcs.euc_kr
];

module.exports.detect = function(buffer) {

  // Tally up the byte occurence statistics.
  var fByteStats = [];
  for (var i = 0; i < 256; i++)
    fByteStats[i] = 0;

  for (var i = buffer.length - 1; i >= 0; i--)
    fByteStats[buffer[i] & 0x00ff]++;

  var fC1Bytes = false;
  for (var i = 0x80; i <= 0x9F; i += 1) {
    if (fByteStats[i] != 0) {
      fC1Bytes = true;
      break;
    }
  }

  var context = {
    fByteStats:  fByteStats,
    fC1Bytes:    fC1Bytes,
    fRawInput:   buffer,
    fRawLength:  buffer.length,
    fInputBytes: buffer,
    fInputLen:   buffer.length
  };

  var match = recognisers.map(function(rec) {
    return rec.match(context);
  }).filter(function(match) {
    return !!match;
  }).sort(function(a, b) {
    return a.confidence - b.confidence;
  }).pop();

  return match ? match.name : null;
};
