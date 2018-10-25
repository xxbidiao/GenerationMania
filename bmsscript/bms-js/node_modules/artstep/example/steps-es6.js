"use strict"

const steps     = require('../')
const expect    = require('chai').expect
const webdriver = require('selenium-webdriver')
const By        = webdriver.By

let driver = new webdriver.Builder().forBrowser('firefox').build()

module.exports = steps()
.When('I visit my website', function*() {
  yield driver.get('http://dt.in.th')
})
.Then('I see $n links below the heading', function*(n) {
  let array = yield driver.findElements(By.css('h1 + ul a'))
  expect(array).to.have.length(+n)
})
.Then('I see a link to "$href"', function*(href) {
  yield driver.findElement(By.css('a[href="' + href + '"]'))
})
.afterAll(function*() {
  yield driver.quit()
})
