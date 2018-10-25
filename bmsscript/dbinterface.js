
module.exports = MongoDBInterface

const MongoClient = require('mongodb').MongoClient;
const assert = require('assert');

// Connection URL
const url = 'mongodb://localhost:27017';

// Database Name
const dbName = 'musicgame';

var mongoClient = null;
var mongoDb = null;

function MongoDBInterface ()
{
    this.client = null;
    this.db = null;
}

init = function(callback)
{
    if(mongoDb == null)
    {
        console.log("Initializing MongoDB...2");
        MongoClient.connect(url, function(err, client) {
        console.log("callback started");
        assert.equal(null, err);
        console.log("Connected successfully to server");
        mongoClient = client;
        mongoDb = client.db(dbName);
        console.log(mongoDb);
        callback();
        });
    }
    else
    {
        console.log("MongoDB is already initialized.");
        callback();
    }
}


close = function()
{
    console.log(this.client);
    mongoClient.close();
    console.log("Connection closed successfully");
}

insertDocuments = function(obj)
{

    if(mongoDb == null)
    {
        init(insertDocumentsReal);
        console.log("aaa");
        console.log(mongoDb);
        insertDocumentsReal(mongoDb,obj,function(){});
    }
    else
    {
        console.log("elseelse");
        insertDocumentsReal(mongoDb,obj,function(){});
    }
}

insertDocumentsReal = function(db, obj, callback) {
  if(db == null)
  {
    console.log("handling...");
    return;
  }
  console.log("normal landing");
  // Get the documents collection
  const collection = db.collection(dbName);
  // Insert some documents
  collection.insertMany([
    obj//{a : 1}, {a : 2}, {a : 3}
  ], function(err, result) {
    assert.equal(err, null);
    assert.equal(3, result.result.n);
    assert.equal(3, result.ops.length);
    console.log("Inserted 3 documents into the collection");
    //callback(result);
  });
}

//init();
insertDocuments(1);
//close();
//i.close();