
module.exports = DataStructure

function DataStructure() {

  var schemas = [].slice.call(arguments)

  function Constructor(object) {
    Constructor.validate(object)
    return object
  }

  Constructor.validate = function(object) {
    for (var i = 0; i < schemas.length; i ++) {
      validate(schemas[i], object)
    }
  }

  return Constructor

}

DataStructure.maybe = function maybe(schema) {
  function MaybeValidator(object) {
    MaybeValidator.validate(object)
    return object
  }
  MaybeValidator.validate = function(value) {
    if (value === null || value === undefined) return
    validate(schema, value)
  }
  return MaybeValidator
}

function validate(schema, value) {
  if (schema === Number) schema = 'number'
  if (schema === String) schema = 'string'
  if (typeof schema === 'string') {
    if (typeof value !== schema) throw new Error('should be a ' + schema)
  } else if (typeof schema === 'function') {
    if (typeof schema.validate === 'function') {
      schema.validate(value)
    } else if (!(value instanceof schema)) {
      throw new Error('should be an instance of ' + schema)
    }
  } else if (typeof schema === 'object') {
    if (!value) throw new Error('should be an object')
    validateObject(schema, value)
  } else {
    throw new Error('invalid schema')
  }
}

function validateObject(schema, object) {
  for (var prop in schema) {
    if (!(prop in object)) {
      throw new Error('missing property: "' + prop + '"')
    }
    try {
      validate(schema[prop], object[prop])
    } catch (e) {
      throw new Error('error in property "' + prop + '": ' + e.message)
    }
  }
}

