module.exports = {
  extends: require.resolve('eslint-config-standard'),
  rules: {
    'comma-dangle': 0,
    'indent': 0,
    'key-spacing': [ 2, { beforeColon: false, afterColon: true, mode: 'minimum' } ],
    'no-multiple-empty-lines': 0,
    'no-multi-spaces': 0,
    'padded-blocks': 0,
    'generator-star-spacing': [ 2, { before: false, after: true } ],
    'spaced-comment': 0,
    'yoda': 0,
  }
}
