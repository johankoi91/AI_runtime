#include "runtime/influxdb_sink.h"

#import <Foundation/Foundation.h>

#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <string>

namespace runtime {

namespace {

std::string env_or_empty(const char* name) {
  const char* value = std::getenv(name);
  return value == nullptr ? std::string{} : std::string(value);
}

std::string escape_tag(std::string value) {
  for (std::size_t index = 0; index < value.size(); ++index) {
    if (value[index] == ',' || value[index] == ' ' || value[index] == '=') {
      value.insert(index, 1, '\\');
      ++index;
    }
  }
  return value;
}

std::string escape_field(std::string value) {
  std::string escaped;
  escaped.reserve(value.size() + 8);
  for (const char value_char : value) {
    if (value_char == '\\' || value_char == '"') {
      escaped.push_back('\\');
    }
    if (value_char == '\n' || value_char == '\r') {
      escaped.push_back(' ');
    } else {
      escaped.push_back(value_char);
    }
  }
  return escaped;
}

std::string make_run_id() {
  const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::system_clock::now().time_since_epoch());
  return "metal-" + std::to_string(now.count());
}

std::string add_query_parameter(std::string url, const char* key, const std::string& value) {
  if (url.find('?') == std::string::npos) {
    url += '?';
  } else {
    url += '&';
  }
  url += key;
  url += '=';
  for (const char value_char : value) {
    if (value_char == ' ') {
      url += "%20";
    } else {
      url.push_back(value_char);
    }
  }
  return url;
}

void send_batch(
    const std::string& base_url,
    const std::string& org,
    const std::string& bucket,
    const std::string& token,
    const std::string& lines) {
  if (lines.empty()) {
    return;
  }
  std::string endpoint = add_query_parameter(base_url + "/api/v2/write", "org", org);
  endpoint = add_query_parameter(endpoint, "bucket", bucket);
  endpoint = add_query_parameter(endpoint, "precision", "ns");
  NSMutableURLRequest* request = [NSMutableURLRequest requestWithURL:
      [NSURL URLWithString:[NSString stringWithUTF8String:endpoint.c_str()]]];
  request.HTTPMethod = @"POST";
  [request setValue:[NSString stringWithFormat:@"Token %s", token.c_str()]
      forHTTPHeaderField:@"Authorization"];
  [request setValue:@"text/plain; charset=utf-8" forHTTPHeaderField:@"Content-Type"];
  request.HTTPBody = [NSData dataWithBytes:lines.data() length:lines.size()];

  NSError* error = nil;
  NSHTTPURLResponse* response = nil;
  [NSURLConnection sendSynchronousRequest:request returningResponse:&response error:&error];
  if (error != nil || response.statusCode >= 300) {
    std::fprintf(
        stderr,
        "[trace] InfluxDB batch write failed: %s\n",
        error == nil ? "HTTP error" : error.localizedDescription.UTF8String);
  }
}

}  // namespace

struct InfluxDbSink::Impl {
  std::string url = env_or_empty("INFLUXDB_URL");
  std::string org = env_or_empty("INFLUXDB_ORG");
  std::string bucket = env_or_empty("INFLUXDB_BUCKET");
  std::string token = env_or_empty("INFLUXDB_TOKEN");
  std::string run_id = env_or_empty("RUNTIME_RUN_ID");
  std::string pending_lines;
  std::size_t sequence = 0;
  std::int64_t last_timestamp = 0;
  bool warned = false;
};

InfluxDbSink::InfluxDbSink() : impl_(std::make_unique<Impl>()) {
  if (impl_->run_id.empty()) {
    impl_->run_id = make_run_id();
  }
  if (!impl_->url.empty() && impl_->url.back() == '/') {
    impl_->url.pop_back();
  }
}

InfluxDbSink::~InfluxDbSink() {
  if (enabled()) {
    send_batch(impl_->url, impl_->org, impl_->bucket, impl_->token, impl_->pending_lines);
  }
}

bool InfluxDbSink::enabled() const {
  return !impl_->url.empty() && !impl_->org.empty() && !impl_->bucket.empty() &&
      !impl_->token.empty();
}

const std::string& InfluxDbSink::run_id() const {
  return impl_->run_id;
}

void InfluxDbSink::emit(
    std::string_view stage,
    std::string_view event,
    std::string_view detail,
    std::size_t bytes,
    double duration_ms) {
  if (!enabled()) {
    if (!impl_->warned) {
      std::fprintf(
          stderr,
          "[trace] InfluxDB disabled; set INFLUXDB_URL, INFLUXDB_ORG, "
          "INFLUXDB_BUCKET and INFLUXDB_TOKEN\n");
      impl_->warned = true;
    }
    return;
  }

  auto timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count();
  if (timestamp <= impl_->last_timestamp) {
    timestamp = impl_->last_timestamp + 1;
  }
  impl_->last_timestamp = timestamp;
  ++impl_->sequence;
  std::string line = "runtime_trace,stage=" + escape_tag(std::string(stage)) +
      ",run_id=" + escape_tag(impl_->run_id) +
      " event=\"" + escape_field(std::string(event)) +
      "\",detail=\"" + escape_field(std::string(detail)) +
      "\",sequence=" + std::to_string(impl_->sequence) + "i" +
      ",bytes=" + std::to_string(bytes) + "i" +
      ",duration_ms=" + std::to_string(duration_ms) + " " +
      std::to_string(timestamp);

  impl_->pending_lines += line;
  impl_->pending_lines.push_back('\n');
}

}  // namespace runtime
