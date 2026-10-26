#include "runtime/influxdb_sink.h"

#import <Foundation/Foundation.h>

#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <string>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <utility>
#include <algorithm>

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

std::string send_batch(
    const std::string& base_url,
    const std::string& org,
    const std::string& bucket,
    const std::string& token,
    const std::string& lines) {
  if (lines.empty()) {
    return {};
  }
  std::string endpoint = add_query_parameter(base_url + "/api/v2/write", "org", org);
  endpoint = add_query_parameter(endpoint, "bucket", bucket);
  endpoint = add_query_parameter(endpoint, "precision", "ns");
  NSMutableURLRequest* request = [NSMutableURLRequest requestWithURL:
      [NSURL URLWithString:[NSString stringWithUTF8String:endpoint.c_str()]]];
  request.HTTPMethod = @"POST";
  request.timeoutInterval = 5.0;
  [request setValue:[NSString stringWithFormat:@"Token %s", token.c_str()]
      forHTTPHeaderField:@"Authorization"];
  [request setValue:@"text/plain; charset=utf-8" forHTTPHeaderField:@"Content-Type"];
  request.HTTPBody = [NSData dataWithBytes:lines.data() length:lines.size()];

  NSError* error = nil;
  NSHTTPURLResponse* response = nil;
  [NSURLConnection sendSynchronousRequest:request returningResponse:&response error:&error];
  if (error != nil) return error.localizedDescription.UTF8String;
  if (response.statusCode < 200 || response.statusCode >= 300)
    return "InfluxDB HTTP " + std::to_string(response.statusCode);
  return {};
}

}  // namespace

struct InfluxDbSink::Impl {
  InfluxDbConfig config;
  mutable std::mutex mutex;
  std::condition_variable wake;
  std::thread worker;
  std::string pending_lines;
  std::size_t pending_events = 0;
  std::size_t in_flight_bytes = 0;
  InfluxDbStatus status;
  std::size_t sequence = 0;
  std::int64_t last_timestamp = 0;
  bool stopping = false;
  bool requested = false;

  void upload() {
    constexpr std::size_t kUploadBatchBytes = 64 * 1024;
    std::unique_lock lock(mutex);
    while (!stopping || !pending_lines.empty()) {
      wake.wait_for(lock, std::chrono::milliseconds(500), [&] {
        return stopping || requested || !pending_lines.empty();
      });
      requested = false;
      if (pending_lines.empty()) continue;
      const std::size_t limit = std::min(kUploadBatchBytes, pending_lines.size());
      std::size_t end = pending_lines.rfind('\n', limit - 1);
      if (end == std::string::npos) end = pending_lines.find('\n');
      if (end == std::string::npos) continue;
      ++end;
      std::string lines = pending_lines.substr(0, end);
      pending_lines.erase(0, end);
      const std::size_t count = static_cast<std::size_t>(
          std::count(lines.begin(), lines.end(), '\n'));
      pending_events -= count;
      in_flight_bytes = lines.size();
      lock.unlock();
      std::string error;
      @autoreleasepool {
        error = send_batch(config.url, config.org, config.bucket, config.token, lines);
      }
      lock.lock();
      in_flight_bytes = 0;
      status.error = error;
      if (error.empty()) {
        status.uploaded_events += count;
      } else if (!stopping) {
        pending_lines.insert(0, std::move(lines));
        pending_events += count;
      } else {
        status.dropped_events += count;
      }
      wake.notify_all();
      if (!error.empty() && !stopping)
        wake.wait_for(lock, std::chrono::seconds(2), [&] { return stopping; });
    }
  }
};

InfluxDbSink::InfluxDbSink(InfluxDbConfig config) : impl_(std::make_unique<Impl>()) {
  auto configure = [](std::string& value, const char* name) {
    const std::string environment = env_or_empty(name);
    if (!environment.empty()) value = environment;
  };
  configure(config.url, "INFLUXDB_URL");
  configure(config.org, "INFLUXDB_ORG");
  configure(config.bucket, "INFLUXDB_BUCKET");
  configure(config.token, "INFLUXDB_TOKEN");
  configure(config.run_id, "RUNTIME_RUN_ID");
  if (config.run_id.empty()) config.run_id = make_run_id();
  while (!config.url.empty() && config.url.back() == '/') config.url.pop_back();
  impl_->config = std::move(config);
  if (enabled()) impl_->worker = std::thread([this] { impl_->upload(); });
}

InfluxDbSink::~InfluxDbSink() {
  {
    std::lock_guard lock(impl_->mutex);
    impl_->stopping = true;
  }
  impl_->wake.notify_all();
  if (impl_->worker.joinable()) impl_->worker.join();
}

bool InfluxDbSink::enabled() const {
  const auto& config = impl_->config;
  return config.enabled && !config.url.empty() && !config.org.empty() && !config.bucket.empty() &&
      !config.token.empty();
}

const std::string& InfluxDbSink::run_id() const {
  return impl_->config.run_id;
}

InfluxDbStatus InfluxDbSink::status() const {
  std::lock_guard lock(impl_->mutex);
  auto status = impl_->status;
  status.pending_bytes = impl_->pending_lines.size() + impl_->in_flight_bytes;
  return status;
}

bool InfluxDbSink::flush(std::chrono::milliseconds timeout) {
  if (!enabled()) return false;
  std::unique_lock lock(impl_->mutex);
  const std::size_t target = impl_->status.accepted_events;
  impl_->requested = true;
  impl_->wake.notify_all();
  const bool complete = impl_->wake.wait_for(lock, timeout, [&] {
    return impl_->status.uploaded_events >= target || impl_->stopping;
  });
  return complete && impl_->status.uploaded_events >= target && impl_->status.dropped_events == 0;
}

void InfluxDbSink::emit(
    std::string_view stage,
    std::string_view event,
    std::string_view detail,
    std::size_t bytes,
    double duration_ms) {
  if (!enabled()) return;
  std::lock_guard lock(impl_->mutex);

  auto timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count();
  if (timestamp <= impl_->last_timestamp) {
    timestamp = impl_->last_timestamp + 1;
  }
  impl_->last_timestamp = timestamp;
  ++impl_->sequence;
  std::string line = "runtime_trace,stage=" + escape_tag(std::string(stage)) +
      ",run_id=" + escape_tag(impl_->config.run_id) +
      " event=\"" + escape_field(std::string(event)) +
      "\",detail=\"" + escape_field(std::string(detail)) +
      "\",sequence=" + std::to_string(impl_->sequence) + "i" +
      ",bytes=" + std::to_string(bytes) + "i" +
      ",duration_ms=" + std::to_string(duration_ms) + " " +
      std::to_string(timestamp);

  if (line.size() + 1 > 64 * 1024 ||
      impl_->pending_lines.size() + impl_->in_flight_bytes + line.size() + 1 >
      impl_->config.max_pending_bytes) {
    ++impl_->status.dropped_events;
    return;
  }
  impl_->pending_lines += line;
  impl_->pending_lines.push_back('\n');
  ++impl_->pending_events;
  ++impl_->status.accepted_events;
  if (impl_->pending_lines.size() >= 256 * 1024) impl_->wake.notify_one();
}

}  // namespace runtime
