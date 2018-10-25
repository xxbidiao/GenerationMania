
async           = require 'asyncawait/async'
await           = require 'asyncawait/await'

steps           = require '../'
{ expect }      = require 'chai'
{ Builder, By } = require 'selenium-webdriver'

{ When, Then, AfterAll } = module.exports = steps()
driver = new Builder().forBrowser('firefox').build()

When 'I visit my website', ->
  driver.get('http://dt.in.th')

Then 'I see $n links below the heading', async (n) ->
  array = await driver.findElements(By.css('h1 + ul a'))
  expect(array).to.have.length(+n)

Then 'I see a link to "$href"', (href) ->
  driver.findElement(By.css "a[href='#{href}']")

AfterAll ->
  driver.quit()
