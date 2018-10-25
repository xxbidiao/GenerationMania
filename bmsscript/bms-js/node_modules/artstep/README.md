
Artstep
=======

Fluent, painless, and beautiful synchronous/asynchronous step definitions for [Cucumber.js](https://github.com/cucumber/cucumber-js).
Supports promises, generators, and async functions, powered by [co](https://www.npmjs.com/package/co).

Oh, and listen to some [nice chill trap and artstep](https://www.youtube.com/watch?v=eKJJqu6P4IQ).


Why?
----

When creating step definitions for [cucumber-js](https://github.com/cucumber/cucumber-js), I always feel like this:

![Y U NO...](http://i0.kym-cdn.com/photos/images/newsfeed/000/089/665/tumblr_l96b01l36p1qdhmifo1_500.jpg)

It simply sucks. As of v0.12:

- If you forget to call the `callback` function, the event queue is drained and Cucumber simply exits without reporting anything.
- No support for promises or generators.
- Having to use `this.` every time to define a single step.
- Regular expressions are ugly.

So I created some wrapper functions for them and then turn it into a library.


API
---

### var steps = require('artstep')

Returns a function that can be exported as a Cucumber steps definitions.

The returned function also has several methods,
which can be used to define steps.
These methods are all _fluent_,
meaning that they all return the same function.


### Synopsis

```js
var steps = require('artstep')

module.exports = steps()
.Given(pattern, handler)
.When(pattern, handler)
.Then(pattern, handler)
.before(...tags, handler)
.after(...tags, handler)
.beforeAll(handler)
.afterAll(handler)
.around(handler)
```

__Note:__ Both PascalCase and lowerCamelCase versions are available for all these methods.


### Handler Function

A handler is:

- A synchronous function. Handler will finish running immediately:

    ```js
    .Then('the title contains "..."', function(text) {
      expect(this.subject.title).to.contain(text)
    })
    ```

- A function that returns a Promise:

    ```js
    .When('I select the $n song', function(n) {
      n = parseInt(n, 10)
      return driver.findElements(By.css('ul.music-list > li'))
      .then(function(items) {
        return items[n - 1].click()
      })
    })
    ```

    In CoffeeScript, it looks very beautiful:

    ```js
    .When 'I select the $n song', (n) ->
      n = parseInt(n, 10)
      driver.findElements(By.css('ul.music-list > li')).then (items) ->
        items[n - 1].click()
    ```

- An ES6 generator function:

    ```js
    .When('I select the $n song', function*(n) {
      n = parseInt(n, 10)
      var items = yield driver.findElements(By.css('ul.music-list > li'))
      yield items[n - 1].click()
    })
    ```

    You can yield promises, generators, thunks, arrays, or objects.
    Then the resolved value will be given back unto you.
    This is possible because of the amazing [co](https://www.npmjs.com/package/co) library.


- An ES7 asynchronous function:

    ```js
    .When('I select the $n song', async function(n) {
      n = parseInt(n, 10)
      var items = await driver.findElements(By.css('ul.music-list > li'))
      await items[n - 1].click()
    })
    ```

- An async function from [asyncawait](https://github.com/yortus/asyncawait),
  which makes this CoffeeScript steps definitions extremely beautiful!

    ```coffee
    .When 'I select the $n song', async ->
      n = parseInt(n, 10)
      items = await driver.findElements(By.css('ul.music-list > li'))
      await items[n - 1].click()
    ```

- `null`, `false` or `undefined`. In this case, the steps will be marked as "pending."


### Around Hooks

The semantic of __around__ hooks changed a little bit.
It passes a function that, when called, runs the scenario,
and returns a Promise which will be resolved when the scenario
finishes executing.

That promise can be yielded.

- ES5 + Promises:

    ```js
    .Around(function(run) {
      return stuffBefore()
      .then(run)
      .then(stuffAfter)
    })
    ```
- ES6 Generators:

    ```js
    .Around(function*(run) {
      var start = Date.now()
      yield run()
      var finish = Date.now()
      console.log('It took', finish - start, 'ms')
    })
    ```

- ES7 Async Functions:

    ```js
    .Around(async function(run) {
      var start = Date.now()
      await run()
      var finish = Date.now()
      console.log('It took', finish - start, 'ms')
    })
    ```
