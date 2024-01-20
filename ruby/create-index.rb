#
# Create a pod-based Pinecone index
#

require 'json'
require 'uri'
require 'net/http'

url = URI("https://api.pinecone.io/indexes")

http = Net::HTTP.new(url.host, url.port)
http.use_ssl = true

request = Net::HTTP::Post.new(url)
request["accept"] = 'application/json'
request["content-type"] = 'application/json'
request["Api-Key"] = '0a86091c-a5b2-4ac0-8bff-a7bf4449feae'

create-index-body = {
  name: 'my-index',
  dimension: 1536,
  metric: 'cosine',
  spec: { 
    pod: { 
      environment: 'us-west1-gcp', 
      pod_type: 'p1.x1'
    }
  }
}

request.body = JSON.generate(create-index-body)
response = http.request(request)
puts response.read_body