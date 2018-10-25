
var steps = require('../')
var webdriver = require('selenium-webdriver')
var By = webdriver.By

var chai = require('chai')
var expect = chai.expect
chai.use(require('chai-as-promised'))

var driver = new webdriver.Builder().forBrowser('firefox').build()

module.exports = steps()
.When('I visit my website', function() {
  return driver.get('http://dt.in.th')
})
.Then('I see $n links below the heading', function(n) {
  return expect(driver.findElements(By.css('h1 + ul a')))
               .to.eventually.have.length(+n)
})
.Then('I see a link to "$href"', function(href) {
  return driver.findElement(By.css('a[href="' + href + '"]'))
})
.afterAll(function() {
  return driver.quit()
})
