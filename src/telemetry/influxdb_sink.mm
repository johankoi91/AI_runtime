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
#include <array>
#include <atomic>
#include <cstring>
#include <vector>
#include <pthread.h>
#include <sys/qos.h>

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
  struct Event {
    std::array<char, 64> stage{};
    std::array<char, 64> event{};
    std::array<char, 512> detail{};
    std::size_t stage_size = 0;
    std::size_t event_size = 0;
    std::size_t detail_size = 0;
    std::size_t bytes = 0;
    std::size_t sequence = 0;
    std::int64_t timestamp = 0;
    double duration_ms = 0;
  };
  InfluxDbConfig config;
  std::mutex queue_mutex;
  std::vector<Event> events;
  std::size_t head = 0;
  std::size_t tail = 0;
  std::size_t count = 0;
  std::atomic<std::size_t> accepted{0};
  std::atomic<std::size_t> uploaded{0};
  std::atomic<std::size_t> dropped{0};
  std::atomic<std::size_t> outstanding{0};
  mutable std::mutex status_mutex;
  std::condition_variable wake;
  std::string error;
  std::atomic_bool stopping{false};

  bool pop(Event& event) {
    std::lock_guard lock(queue_mutex);
    if (count == 0) return false;
    event = events[head];
    head = (head + 1) % events.size();
    --count;
    return true;
  }

  std::string encode(const Event& event, std::int64_t& last_timestamp) const {
    const auto timestamp = std::max(event.timestamp, last_timestamp + 1);
    last_timestamp = timestamp;
    return "runtime_trace,stage=" + escape_tag(std::string(event.stage.data(), event.stage_size)) +
        ",run_id=" + escape_tag(config.run_id) +
        " event=\"" + escape_field(std::string(event.event.data(), event.event_size)) +
        "\",detail=\"" + escape_field(std::string(event.detail.data(), event.detail_size)) +
        "\",sequence=" + std::to_string(event.sequence) + "i,bytes=" +
        std::to_string(event.bytes) + "i,duration_ms=" + std::to_string(event.duration_ms) +
        " " + std::to_string(timestamp) + "\n";
  }

  void upload() {
    std::int64_t last_timestamp = 0;
    auto stop_deadline = std::chrono::steady_clock::time_point::max();
    std::string lines;
    std::size_t batch_events = 0;
    Event event;
    while (true) {
      if (stopping.load() && stop_deadline == std::chrono::steady_clock::time_point::max())
        stop_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
      if (std::chrono::steady_clock::now() >= stop_deadline) break;
      while (lines.size() < 60 * 1024 && pop(event)) {
        std::string line = encode(event, last_timestamp);
        if (line.size() > 4 * 1024) {
          dropped.fetch_add(1);
          outstanding.fetch_sub(1);
          continue;
        }
        lines += line;
        ++batch_events;
      }
      if (lines.empty()) {
        if (stopping.load()) break;
        std::unique_lock lock(status_mutex);
        wake.wait_for(lock, std::chrono::milliseconds(10));
        continue;
      }
      std::string upload_error;
      @autoreleasepool {
        upload_error = send_batch(config.url, config.org, config.bucket, config.token, lines);
      }
      {
        std::lock_guard lock(status_mutex);
        error = upload_error;
        if (upload_error.empty()) {
          uploaded.fetch_add(batch_events);
          outstanding.fetch_sub(batch_events);
          lines.clear();
          batch_events = 0;
        }
      }
      wake.notify_all();
      if (!upload_error.empty()) {
        if (stopping.load()) break;
        std::unique_lock lock(status_mutex);
        wake.wait_for(lock, std::chrono::seconds(2), [&] { return stopping.load(); });
      }
    }
    {
      std::lock_guard lock(queue_mutex);
      stopping.store(true);
      dropped.fetch_add(outstanding.exchange(0));
    }
    wake.notify_all();
  }

  void fail() {
    std::lock_guard lock(status_mutex);
    std::lock_guard queue_lock(queue_mutex);
    error = "trace upload worker failed";
    dropped.fetch_add(outstanding.exchange(0));
    stopping.store(true);
    wake.notify_all();
  }

  void start(const std::shared_ptr<Impl>& state) {
    std::thread([state] {
      pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);
      try {
        state->upload();
      } catch (...) {
        state->fail();
      }
    }).detach();
  }

  void enqueue(std::string_view stage, std::string_view name, std::string_view detail,
               std::size_t bytes, double duration_ms) noexcept {
    if (stopping.load() || stage.size() > Event{}.stage.size() ||
        name.size() > Event{}.event.size() || detail.size() > Event{}.detail.size()) {
      dropped.fetch_add(1);
      return;
    }
    std::unique_lock lock(queue_mutex, std::try_to_lock);
    if (!lock.owns_lock() || stopping.load() || outstanding.load() >= events.size()) {
      dropped.fetch_add(1);
      return;
    }
    Event& event = events[tail];
    if (!stage.empty()) std::memcpy(event.stage.data(), stage.data(), stage.size());
    if (!name.empty()) std::memcpy(event.event.data(), name.data(), name.size());
    if (!detail.empty()) std::memcpy(event.detail.data(), detail.data(), detail.size());
    event.stage_size = stage.size();
    event.event_size = name.size();
    event.detail_size = detail.size();
    event.bytes = bytes;
    event.duration_ms = duration_ms;
    event.timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    event.sequence = accepted.fetch_add(1) + 1;
    outstanding.fetch_add(1);
    tail = (tail + 1) % events.size();
    ++count;
  }
};

InfluxDbSink::InfluxDbSink(InfluxDbConfig config) : impl_(std::make_shared<Impl>()) {
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
  if (enabled()) {
    impl_->events.resize(impl_->config.max_pending_bytes / sizeof(Impl::Event));
    impl_->start(impl_);
  }
}

InfluxDbSink::~InfluxDbSink() {
  impl_->stopping.store(true);
  impl_->wake.notify_all();
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
  InfluxDbStatus status;
  status.accepted_events = impl_->accepted.load();
  status.uploaded_events = impl_->uploaded.load();
  status.dropped_events = impl_->dropped.load();
  status.pending_bytes = impl_->outstanding.load() * sizeof(Impl::Event);
  std::lock_guard lock(impl_->status_mutex);
  status.error = impl_->error;
  return status;
}

bool InfluxDbSink::flush(std::chrono::milliseconds timeout) {
  if (!enabled()) return false;
  const std::size_t target = impl_->accepted.load();
  std::unique_lock lock(impl_->status_mutex);
  impl_->wake.notify_all();
  const bool complete = impl_->wake.wait_for(lock, timeout, [&] {
    return impl_->uploaded.load() >= target || impl_->stopping.load();
  });
  return complete && impl_->uploaded.load() >= target && impl_->dropped.load() == 0;
}

void InfluxDbSink::emit(
    std::string_view stage,
    std::string_view event,
    std::string_view detail,
    std::size_t bytes,
    double duration_ms) noexcept {
  if (!enabled()) return;
  impl_->enqueue(stage, event, detail, bytes, duration_ms);
}

}  // namespace runtime
