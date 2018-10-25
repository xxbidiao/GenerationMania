
var steps = require('./')

var already = false

module.exports = steps()
.given('Artstep is installed', function() {
  /* It is! And look, no callback! */
})
.when('I use it', function() {
  return new Promise(function(resolve, reject) {
    /* I am busy using it... */
    setTimeout(resolve, 300)
  })
})
.when('the function is not given')
.when('the function returns PENDING', function() {
  return this.PENDING
})
.then('I am "$adjective"', function(x) {
  setTimeout(function() {
    console.log('\033[1;46;22;30m I\'M SO ' + x.toUpperCase() + '! \033[m')
  }, 100)
})
.then('the scenario is pending', function(x) {
  throw new Error('Scenario is pending! This step must not run!')
})
.before(function() {
  console.log('before')
})
.after(function() {
  console.log('after')
})
.around(function(run) {
  var start = Date.now()
  return run().then(function() {
    var finish = Date.now()
    console.log('takes', finish - start, 'ms')
  })
})
.afterAll(function() {
  console.log('all is done...\n')
})
.beforeAll('@Test', function() {
  if (already) {
    throw new Error('wtf')
  } else {
    already = true
    console.log('before all @Test')
  }
})
.before('@Test', function() {
  this.taggedBeforeHook = true
})
.then('the tagged before hook is run', function() {
  if (!this.taggedBeforeHook) throw new Error('Ouch it did not run')
})
.then('the tagged before hook is not run', function() {
  if (this.taggedBeforeHook) throw new Error('Ouch it should not run')
})
.beforeAll(function() {
  console.log('Get ready!')
})
