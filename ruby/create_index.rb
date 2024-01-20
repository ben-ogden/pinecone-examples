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
request["Api-Key"] = 'CHANGEME'

create_index_body = {
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

request.body = JSON.generate(create_index_body)
response = http.request(request)
puts response.read_body