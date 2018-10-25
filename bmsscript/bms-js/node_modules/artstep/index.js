
require('es6-promise').polyfill()

var co = require('co')

var API = {
  given: function(pattern, callback) {
    this.Given(pattern, wrap(callback))
  },
  when: function(pattern, callback) {
    this.When(pattern, wrap(callback))
  },
  then: function(pattern, callback) {
    this.Then(pattern, wrap(callback))
  },
  before: function(callback) {
    var args = [].slice.call(arguments)
    this.Before.apply(this, args.concat([wrap(args.pop())]))
  },
  after: function(callback) {
    var args = [].slice.call(arguments)
    this.After.apply(this, args.concat([wrap(args.pop())]))
  },
  afterAll: function(callback) {
    this.registerHandler('AfterFeatures', wrap(callback))
  },
  beforeAll: function(callback) {
    var args = [].slice.call(arguments)
    this.Before.apply(this, args.concat([wrap(once(args.pop()))]))
  },
  hook: function(name, callback) {
    this.registerHandler(name, wrap(callback))
  },
  around: function(callback) {
    var finish  = function() {
      throw new Error('Must yield/call run().')
    }
    var wrapped = wrap(callback)
    this.Around(function(run) {
      wrapped(function() {
        return new Promise(function(resolve, reject) {
          run(function(fin) {
            finish = fin
            resolve()
          })
        })
      }, function() {
        finish()
      })
    })
  },
}

var PENDING = {}

module.exports = function steps() {
  var hooks = []
  function artstepDefinitionsWrapper() {
    for (var i = 0; i < hooks.length; i ++) hooks[i].call(this)
  }
  function addAPI(key, f) {
    artstepDefinitionsWrapper[key] =
    artstepDefinitionsWrapper[key.charAt(0).toUpperCase() + key.substr(1)] =
    function() {
      var args = [].slice.call(arguments)
      hooks.push(function() {
        f.apply(this, args)
      })
      return this
    }
  }
  for (var key in API) {
    if (Object.hasOwnProperty.call(API, key)) {
      addAPI(key, API[key])
    }
  }
  return artstepDefinitionsWrapper
}

exports.PENDING = PENDING

function wrap(fn) {
  fn      = fn || pending
  return function() {
    var callback  = arguments[arguments.length - 1]
    var world     = this
    var args      = [].slice.call(arguments)
    co(function() {
      world.PENDING = PENDING
      return fn.apply(world, args)
    })
    .then(
      function(result) {
        if (result === PENDING) {
          callback.pending()
        } else {
          callback()
        }
      },
      function(err) {
        callback(err)
      }
    )
  }
}

function pending() {
  return PENDING
}

function once(f) {
  var run = false
  var result
  return function() {
    if (!run) {
      run = true
      result = f.apply(this, arguments)
    }
    return result
  }
}
