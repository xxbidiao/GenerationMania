
data-structure
==============

This small library lets you declare a custom data structure.

```javascript
var DataStructure = require('data-structure')
var Event = new DataStructure({
  position:  Number,
  beat:      Number,
  time:      Number,
})
```

Usage

```javascript
var event = new Event({ position: 1, beat: 1, time: 0.5 })
```

Why?
----

Well, sometimes you create an object, that's fine...

```javascript
var event = { position: 1, beat: 1, time: 0.5 }
```

But when you pass it around in your application:

```javascript
function processEvent(event) {
  findObjectsNear(event.position)
}
```

Users of your class may wonder what an "event" look like.
Some solutions exist:

- Use comments to describe the structure of the object.
    - May quickly be out of date!
- Create an opaque class to hold values.
    - Maintenance burden!
- Use Flow/TypeScript
    - Now what if we're using 6to5 or CoffeeScript?
- __Use this library!!__
    - Now your code is self-documenting.
    - The returned function just validates and returns passed object.
        - Code becomes self-checking code!
        - No extra type to mess with!
