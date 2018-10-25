
var assert = require('assert')
var DataStructure = require('./')

console.log('Let\'s test this thing!!')

var Event = new DataStructure({
      beat:     Number,
      position: Number,
      time:     Number,
    })

bail('When not sending object', function() {
  new Event()
}, /should be an object/i)

bail('When the "beat" attribute is missing', function() {
  new Event({ position: 1, time: 1 })
}, /missing property: "beat"/i)

test('When passing a valid object', function(ok) {
  var object = { position: 1, beat: 2, time: 3 }
  assert.strictEqual(new Event(object), object)
  ok('that same object is returned')
})

bail('When passing a string in place of number', function(ok) {
  new Event({ position: 'wow', time: 1, beat: 3 })
}, /should be a number/i)

bail('When passing an object in place of number', function(ok) {
  new Event({ position: { }, time: 1, beat: 3 })
}, /should be a number/i)

bail('When passing an object in place of string', function(ok) {
  var V = new DataStructure({ x: String })
  new V({ x: { } })
}, /should be a string/i)

section('Data Inheritance', function() {
  var NoteEvent = new DataStructure(Event, { channel: String })
  bail('When missing parent prop', function() {
    new NoteEvent({ channel: 'green', beat: 1, time: 1 })
  }, /position/)
  bail('When missing child prop', function() {
    new NoteEvent({ position: 1, beat: 1, time: 1 })
  }, /channel/)
})

section('DataStructure.maybe', function() {
  var NoteEvent = new DataStructure({ channel: DataStructure.maybe(String) })
  bail('When not provided', function() {
    new NoteEvent({ })
  }, /missing/)
  test('When undefined', function(ok) {
    new NoteEvent({ channel: undefined })
    ok('That is ok!')
  })
  test('When null', function(ok) {
    new NoteEvent({ channel: null })
    ok('That is ok!')
  })
  test('When type is correct', function(ok) {
    new NoteEvent({ channel: 'wow' })
    ok('That is great!')
  })
  bail('When type is incorrect', function() {
    new NoteEvent({ channel: 5 })
  }, /string/)
})

console.log("\033[1;32m素晴らしい！\033[0m")

function section(name, f) {
  console.log('============================================')
  console.log(name)
  console.log('============================================')
  f()
}

function bail(when, f, regex) {
  test(when, function(ok) {
    var error = null
    try {
      f()
      error = 'Expected ' + f + ' to throw error!'
    } catch (e) {
      if (regex && !regex.test(e.message)) {
        error = 'Expected ' + e.message + ' to match ' + regex
      } else {
        ok('it throws "' + e.message + '"')
      }
    }
    if (error !== null) {
      throw new Error(error)
    }
  })
}

function test(description, callback) {
  console.log(' + ' + description)
  callback(function ok(text) {
    console.log(' |--> ' + text)
  })
  console.log('')
}
