#import <AVFoundation/AVFoundation.h>
#import <AppKit/AppKit.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "runtime/cpu_backend.h"
#include "runtime/hush_model.h"
#include "runtime/hush_session.h"
#include "runtime/metal_backend.h"

namespace {

struct Metrics {
  std::atomic<std::uint64_t> frames{0};
  std::atomic<double> latency_ms{0.0};
  std::atomic<double> input_rms{0.0};
  std::atomic<double> output_rms{0.0};
  std::mutex waveform_mutex;
  std::vector<float> waveform;
  std::vector<float> output_waveform;
  std::string error;
};

}

@interface WaveformView : NSView
@property(nonatomic, assign) Metrics* metrics;
@property(nonatomic, assign) BOOL output;
@end

@implementation WaveformView
- (void)drawRect:(NSRect)rect {
  [[NSColor colorWithRed:0.04 green:0.06 blue:0.10 alpha:1.0] setFill];
  NSRectFill(rect);
  std::vector<float> values;
  {
    std::lock_guard<std::mutex> lock(self.metrics->waveform_mutex);
    values = self.output ? self.metrics->output_waveform : self.metrics->waveform;
  }
  if (values.empty()) return;
  NSBezierPath* path = [NSBezierPath bezierPath];
  const CGFloat mid = NSMidY(rect);
  const CGFloat scale = rect.size.height * 0.42;
  for (std::size_t index = 0; index < values.size(); ++index) {
    const CGFloat x = rect.origin.x + rect.size.width * index / std::max<std::size_t>(1, values.size() - 1);
    const CGFloat y = mid - std::max(-1.0f, std::min(1.0f, values[index])) * scale;
    if (index == 0) [path moveToPoint:NSMakePoint(x, y)];
    else [path lineToPoint:NSMakePoint(x, y)];
  }
  if (self.output) [[NSColor colorWithRed:0.35 green:0.95 blue:0.55 alpha:1.0] setStroke];
  else [[NSColor colorWithRed:0.25 green:0.82 blue:0.95 alpha:1.0] setStroke];
  [path setLineWidth:1.5];
  [path stroke];
}
@end

@interface AppDelegate : NSObject <NSApplicationDelegate>
@end

@implementation AppDelegate {
  NSWindow* window_;
  NSTextField* model_label_;
  NSTextField* status_label_;
  NSTextField* metrics_label_;
  NSTextField* demo_label_;
  NSButton* record_button_;
  AVAudioPlayer* player_;
  AVAudioConverter* converter_;
  AVAudioFormat* capture_format_;
  AVAudioFile* native_capture_file_;
  std::atomic_bool recording_;
  std::mutex recording_mutex_;
  std::vector<float> raw_recording_;
  std::vector<float> bypass_recording_;
  std::vector<float> enhanced_recording_;
  std::vector<float> pending_audio_;
  dispatch_queue_t recording_queue_;
  NSButton* start_button_;
  NSPopUpButton* backend_popup_;
  WaveformView* waveform_view_;
  WaveformView* output_waveform_view_;
  AVAudioEngine* engine_;
  Metrics metrics_;
  std::unique_ptr<runtime::Backend> backend_;
  std::unique_ptr<runtime::HushModel> model_;
  std::unique_ptr<runtime::HushStreamSession> session_;
  std::unique_ptr<runtime::MemoryPool> pool_;
}

- (void)applicationDidFinishLaunching:(NSNotification*)notification {
  (void)notification;
  recording_.store(false);
  [self buildWindow];
  const char* env_path = std::getenv("HUSH_MODEL_BUNDLE");
  if (env_path != nullptr) {
    [self loadModel:[NSString stringWithUTF8String:env_path]];
  } else {
    NSString* bundled_model = [[NSBundle mainBundle] pathForResource:@"hush-real-model" ofType:@"tar.gz"];
    if (bundled_model != nil) [self loadModel:bundled_model];
  }
}

- (void)buildWindow {
  window_ = [[NSWindow alloc] initWithContentRect:NSMakeRect(0, 0, 920, 610)
                                          styleMask:(NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                                                      NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable)
                                            backing:NSBackingStoreBuffered defer:NO];
  [window_ setTitle:@"Hush Metal Runtime Lab"];
  [window_ center];
  NSView* content = window_.contentView;
  [content setWantsLayer:YES];
  content.layer.backgroundColor = [NSColor colorWithRed:0.025 green:0.035 blue:0.06 alpha:1].CGColor;

  NSTextField* title = [NSTextField labelWithString:@"Hush / Metal Runtime"];
  title.font = [NSFont systemFontOfSize:26 weight:NSFontWeightSemibold];
  title.textColor = [NSColor whiteColor];
  title.frame = NSMakeRect(32, 545, 450, 34);
  [content addSubview:title];
  NSTextField* subtitle = [NSTextField labelWithString:@"Real-time speech enhancement on Apple Silicon"];
  subtitle.textColor = [NSColor secondaryLabelColor];
  subtitle.frame = NSMakeRect(34, 520, 500, 22);
  [content addSubview:subtitle];

  model_label_ = [NSTextField labelWithString:@"Model: not loaded"];
  model_label_.textColor = [NSColor colorWithRed:0.55 green:0.82 blue:0.92 alpha:1];
  model_label_.frame = NSMakeRect(32, 480, 680, 22);
  [content addSubview:model_label_];
  NSButton* choose = [NSButton buttonWithTitle:@"Choose Model…" target:self action:@selector(chooseModel:)];
  choose.frame = NSMakeRect(735, 474, 145, 32);
  [content addSubview:choose];

  NSTextField* backend_label = [NSTextField labelWithString:@"Backend"];
  backend_label.textColor = [NSColor secondaryLabelColor];
  backend_label.frame = NSMakeRect(32, 438, 80, 22);
  [content addSubview:backend_label];
  backend_popup_ = [[NSPopUpButton alloc] initWithFrame:NSMakeRect(105, 434, 120, 28) pullsDown:NO];
  [backend_popup_ addItemsWithTitles:@[@"Metal", @"CPU"]];
  [content addSubview:backend_popup_];
  start_button_ = [NSButton buttonWithTitle:@"Start Microphone" target:self action:@selector(toggle:)];
  start_button_.bezelStyle = NSBezelStyleRounded;
  start_button_.frame = NSMakeRect(245, 432, 155, 32);
  [content addSubview:start_button_];
  status_label_ = [NSTextField labelWithString:@"Load a Hush model to begin."];
  status_label_.textColor = [NSColor secondaryLabelColor];
  status_label_.frame = NSMakeRect(420, 438, 430, 22);
  [content addSubview:status_label_];

  waveform_view_ = [[WaveformView alloc] initWithFrame:NSMakeRect(32, 300, 848, 105)];
  waveform_view_.metrics = &metrics_;
  waveform_view_.output = NO;
  waveform_view_.wantsLayer = YES;
  waveform_view_.layer.cornerRadius = 10;
  [content addSubview:waveform_view_];
  NSTextField* wave_title = [NSTextField labelWithString:@"INPUT  ·  microphone waveform"];
  wave_title.textColor = [NSColor secondaryLabelColor];
  wave_title.frame = NSMakeRect(48, 385, 400, 22);
  [content addSubview:wave_title];
  output_waveform_view_ = [[WaveformView alloc] initWithFrame:NSMakeRect(32, 165, 848, 105)];
  output_waveform_view_.metrics = &metrics_;
  output_waveform_view_.output = YES;
  output_waveform_view_.wantsLayer = YES;
  output_waveform_view_.layer.cornerRadius = 10;
  [content addSubview:output_waveform_view_];
  NSTextField* output_title = [NSTextField labelWithString:@"OUTPUT  ·  Hush enhanced waveform"];
  output_title.textColor = [NSColor colorWithRed:0.35 green:0.95 blue:0.55 alpha:1.0];
  output_title.frame = NSMakeRect(48, 275, 400, 22);
  [content addSubview:output_title];

  metrics_label_ = [NSTextField labelWithString:@"Frames 0    Latency —    Input RMS —    Output RMS —"];
  metrics_label_.font = [NSFont monospacedSystemFontOfSize:13 weight:NSFontWeightRegular];
  metrics_label_.textColor = [NSColor colorWithRed:0.65 green:0.86 blue:0.72 alpha:1];
  metrics_label_.frame = NSMakeRect(32, 112, 848, 28);
  [content addSubview:metrics_label_];
  NSTextField* hint = [NSTextField labelWithString:@"Input: 16 kHz mono · 160 samples/frame · Metal output path · Permission required"];
  hint.textColor = [NSColor tertiaryLabelColor];
  hint.frame = NSMakeRect(32, 72, 850, 22);
  [content addSubview:hint];
  NSButton* generate_demo = [NSButton buttonWithTitle:@"Generate Background Voice Demo" target:self action:@selector(generateInterferenceDemo:)];
  generate_demo.frame = NSMakeRect(32, 30, 220, 30);
  [content addSubview:generate_demo];
  NSButton* play_input = [NSButton buttonWithTitle:@"Play Mixed Input" target:self action:@selector(playMixedInput:)];
  play_input.frame = NSMakeRect(265, 30, 135, 30);
  [content addSubview:play_input];
  NSButton* play_output = [NSButton buttonWithTitle:@"Play Enhanced Output" target:self action:@selector(playEnhancedOutput:)];
  play_output.frame = NSMakeRect(412, 30, 150, 30);
  [content addSubview:play_output];
  record_button_ = [NSButton buttonWithTitle:@"Record A/B Test" target:self action:@selector(recordComparison:)];
  record_button_.frame = NSMakeRect(32, 0, 180, 28);
  [content addSubview:record_button_];
  NSButton* play_raw = [NSButton buttonWithTitle:@"Play Raw Mic" target:self action:@selector(playRawMic:)];
  play_raw.frame = NSMakeRect(225, 0, 115, 28);
  [content addSubview:play_raw];
  NSButton* play_bypass = [NSButton buttonWithTitle:@"Play Bypass" target:self action:@selector(playBypass:)];
  play_bypass.frame = NSMakeRect(350, 0, 105, 28);
  [content addSubview:play_bypass];
  NSButton* play_hush = [NSButton buttonWithTitle:@"Play Hush" target:self action:@selector(playHush:)];
  play_hush.frame = NSMakeRect(465, 0, 100, 28);
  [content addSubview:play_hush];
  demo_label_ = [NSTextField labelWithString:@"Demo: generate a simulated primary voice + background speaker mix"];
  demo_label_.textColor = [NSColor tertiaryLabelColor];
  demo_label_.frame = NSMakeRect(580, 34, 300, 22);
  [content addSubview:demo_label_];
  [window_ makeKeyAndOrderFront:nil];
  [NSTimer scheduledTimerWithTimeInterval:0.1 target:self selector:@selector(refreshUI:) userInfo:nil repeats:YES];
  recording_queue_ = dispatch_queue_create("com.airuntime.hush.recording", DISPATCH_QUEUE_SERIAL);
}

- (void)recordComparison:(id)sender {
  (void)sender;
  if (!session_) { demo_label_.stringValue = @"A/B test: load the model first"; return; }
  if (engine_ == nil) { demo_label_.stringValue = @"A/B test: start the microphone first"; return; }
  if (!recording_) {
    {
      std::lock_guard<std::mutex> lock(recording_mutex_);
      raw_recording_.clear(); bypass_recording_.clear(); enhanced_recording_.clear();
    }
    recording_.store(true);
    NSError* file_error = nil;
    native_capture_file_ = [[AVAudioFile alloc] initForWriting:
        [NSURL fileURLWithPath:@"/tmp/hush-mic-native.caf"]
        settings:capture_format_.settings error:&file_error];
    if (native_capture_file_ == nil || file_error != nil) {
      recording_.store(false);
      demo_label_.stringValue = file_error.localizedDescription ?: @"Could not open native capture file";
      return;
    }
    record_button_.title = @"Stop A/B Recording";
    demo_label_.stringValue = @"A/B test recording… click Stop when finished";
    return;
  }
  // Stop the audio tap before closing the native file. This makes the file
  // boundary deterministic and prevents the offline reader racing the tap.
  recording_.store(false);
  [engine_.inputNode removeTapOnBus:0];
  [engine_ stop];
  engine_ = nil;
  AVAudioFile* native_file = native_capture_file_;
  native_capture_file_ = nil;
  if (native_file == nil) {
    demo_label_.stringValue = @"A/B test: no native audio was captured";
    return;
  }
  record_button_.title = @"Record A/B Test";
  demo_label_.stringValue = @"A/B test: processing the complete recording…";
  dispatch_async(recording_queue_, ^{
    try {
      const AVAudioFrameCount native_frames = native_file.length;
      if (native_frames == 0) throw std::runtime_error("native capture contains no frames");
      AVAudioPCMBuffer* native_buffer = [[AVAudioPCMBuffer alloc]
          initWithPCMFormat:native_file.processingFormat frameCapacity:native_frames];
      NSError* read_error = nil;
      if (![native_file readIntoBuffer:native_buffer error:&read_error]) {
        throw std::runtime_error(read_error.localizedDescription.UTF8String);
      }
      AVAudioFormat* target_format = [[AVAudioFormat alloc] initStandardFormatWithSampleRate:16000 channels:1];
      AVAudioConverter* offline_converter = [[AVAudioConverter alloc]
          initFromFormat:native_file.processingFormat toFormat:target_format];
      const AVAudioFrameCount target_capacity = static_cast<AVAudioFrameCount>(
          std::ceil(native_frames * 16000.0 / native_file.processingFormat.sampleRate) + 320);
      AVAudioPCMBuffer* target_buffer = [[AVAudioPCMBuffer alloc]
          initWithPCMFormat:target_format frameCapacity:target_capacity];
      __block BOOL delivered = NO;
      if ([offline_converter convertToBuffer:target_buffer error:&read_error withInputFromBlock:^AVAudioBuffer*(AVAudioPacketCount, AVAudioConverterInputStatus* status) {
            if (delivered) { *status = AVAudioConverterInputStatus_EndOfStream; return nil; }
            delivered = YES; *status = AVAudioConverterInputStatus_HaveData; return native_buffer;
          }] == AVAudioConverterOutputStatus_Error || read_error != nil) {
        throw std::runtime_error(read_error.localizedDescription.UTF8String);
      }
      std::vector<float> raw(target_buffer.frameLength);
      std::copy_n(target_buffer.floatChannelData[0], target_buffer.frameLength, raw.data());
      runtime::HushStreamSession offline_session(*self->model_, *self->backend_, {});
      runtime::MemoryPool offline_pool;
      std::vector<float> enhanced; enhanced.reserve(raw.size());
      for (std::size_t offset = 0; offset < raw.size(); offset += 160) {
        std::vector<float> frame(160, 0.0f);
        const std::size_t count = std::min<std::size_t>(160, raw.size() - offset);
        std::copy_n(raw.data() + offset, count, frame.data());
        const auto output = offline_session.process_audio(runtime::AudioFrame::mono_16k(std::move(frame)),
                                                           "feat_erb", "feat_spec", "m", *self->backend_, offline_pool);
        enhanced.insert(enhanced.end(), output.samples().begin(), output.samples().end());
        offline_pool.reset();
      }
      enhanced.resize(raw.size());
      float peak = 0.0f;
      for (float value : enhanced) peak = std::max(peak, std::fabs(value));
      const bool clipped = peak > 0.98f;
      if (clipped) {
        const float scale = 0.98f / peak;
        for (float& value : enhanced) value *= scale;
      }
      std::lock_guard<std::mutex> lock(self->recording_mutex_);
      self->enhanced_recording_ = enhanced;
      self->raw_recording_ = raw;
      self->bypass_recording_ = raw;
      runtime::write_wav_mono16("/tmp/hush-mic-raw.wav", raw);
      runtime::write_wav_mono16("/tmp/hush-mic-bypass.wav", raw);
      runtime::write_wav_mono16("/tmp/hush-mic-enhanced.wav", self->enhanced_recording_);
      const double seconds = static_cast<double>(raw.size()) / 16000.0;
      dispatch_async(dispatch_get_main_queue(), ^{
        self->demo_label_.stringValue = [NSString stringWithFormat:@"A/B ready · %.2fs · %@ peak protected", seconds, clipped ? @"Hush" : @"No"];
        // Resume the live monitor after the complete recording has been
        // materialized. The next tap starts from an empty converter state.
        if (self->engine_ == nil) [self toggle:nil];
      });
    } catch (const std::exception& error) {
      const std::string message = error.what();
      dispatch_async(dispatch_get_main_queue(), ^{
        self->demo_label_.stringValue = [NSString stringWithUTF8String:message.c_str()];
        if (self->engine_ == nil) [self toggle:nil];
      });
    }
  });
}

- (void)generateInterferenceDemo:(id)sender {
  (void)sender;
  if (!model_) { demo_label_.stringValue = @"Demo: load the model first"; return; }
  demo_label_.stringValue = @"Demo: processing simulated background speaker…";
  auto model = model_.get();
  auto backend = backend_.get();
  dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
    try {
      const std::size_t total = 16'000 * 3;
      std::vector<float> mixed(total);
      for (std::size_t index = 0; index < total; ++index) {
        const double time = static_cast<double>(index) / 16'000.0;
        const float primary = 0.18f * static_cast<float>(std::sin(2.0 * M_PI * 180.0 * time + 0.7 * std::sin(2.0 * M_PI * 3.0 * time)));
        const float background = 0.13f * static_cast<float>(std::sin(2.0 * M_PI * 265.0 * time + 1.1 * std::sin(2.0 * M_PI * 2.0 * time)));
        mixed[index] = primary + background;
      }
      runtime::write_wav_mono16("/tmp/hush-background-input.wav", mixed);
      runtime::HushStreamSession demo_session(*model, *backend, {});
      runtime::MemoryPool demo_pool;
      std::vector<float> enhanced; enhanced.reserve(total);
      for (std::size_t offset = 0; offset < total; offset += 160) {
        std::vector<float> frame(mixed.begin() + offset, mixed.begin() + offset + 160);
        const auto output = demo_session.process_audio(runtime::AudioFrame::mono_16k(std::move(frame)),
                                                        "feat_erb", "feat_spec", "m", *backend, demo_pool);
        enhanced.insert(enhanced.end(), output.samples().begin(), output.samples().end());
        demo_pool.reset();
      }
      runtime::write_wav_mono16("/tmp/hush-background-output.wav", enhanced);
      dispatch_async(dispatch_get_main_queue(), ^{ demo_label_.stringValue = @"Demo ready: mixed input and enhanced output saved in /tmp"; });
    } catch (const std::exception& error) {
      const std::string message = error.what();
      dispatch_async(dispatch_get_main_queue(), ^{ demo_label_.stringValue = [NSString stringWithUTF8String:message.c_str()]; });
    }
  });
}

- (void)playMixedInput:(id)sender { (void)sender; [self playDemoFile:@"/tmp/hush-background-input.wav"]; }
- (void)playEnhancedOutput:(id)sender { (void)sender; [self playDemoFile:@"/tmp/hush-background-output.wav"]; }
- (void)playRawMic:(id)sender { (void)sender; [self playDemoFile:@"/tmp/hush-mic-raw.wav"]; }
- (void)playBypass:(id)sender { (void)sender; [self playDemoFile:@"/tmp/hush-mic-bypass.wav"]; }
- (void)playHush:(id)sender { (void)sender; [self playDemoFile:@"/tmp/hush-mic-enhanced.wav"]; }
- (void)playDemoFile:(NSString*)path {
  if (![[NSFileManager defaultManager] fileExistsAtPath:path]) { demo_label_.stringValue = @"Demo: generate the files first"; return; }
  player_ = [[AVAudioPlayer alloc] initWithContentsOfURL:[NSURL fileURLWithPath:path] error:nil];
  [player_ play];
}

- (void)chooseModel:(id)sender {
  (void)sender;
  NSOpenPanel* panel = [NSOpenPanel openPanel];
  panel.allowedFileTypes = @[@"gz"];
  panel.canChooseDirectories = NO;
  if ([panel runModal] == NSModalResponseOK) [self loadModel:panel.URL.path];
}

- (void)loadModel:(NSString*)path {
  try {
    model_ = std::make_unique<runtime::HushModel>(runtime::HushModel::load(path.UTF8String));
    if ([backend_popup_.titleOfSelectedItem isEqualToString:@"Metal"]) backend_ = std::make_unique<runtime::MetalBackend>();
    else backend_ = std::make_unique<runtime::CPUBackend>();
    session_ = std::make_unique<runtime::HushStreamSession>(*model_, *backend_, std::vector<runtime::StateBinding>{});
    pool_ = std::make_unique<runtime::MemoryPool>();
    model_label_.stringValue = [NSString stringWithFormat:@"Model: %@ · 16 kHz · encoder ready", path.lastPathComponent];
    status_label_.stringValue = @"Ready. Press Start Microphone.";
  } catch (const std::exception& error) {
    status_label_.stringValue = [NSString stringWithUTF8String:error.what()];
  }
}

- (void)toggle:(id)sender {
  (void)sender;
  if (engine_ != nil) { [engine_ stop]; [engine_.inputNode removeTapOnBus:0]; engine_ = nil; start_button_.title = @"Start Microphone"; status_label_.stringValue = @"Stopped"; return; }
  if (!session_) { status_label_.stringValue = @"Choose a model first."; return; }
  engine_ = [[AVAudioEngine alloc] init];
  AVAudioInputNode* input = engine_.inputNode;
  AVAudioFormat* output_format = [[AVAudioFormat alloc] initStandardFormatWithSampleRate:16000 channels:1];
  converter_ = [[AVAudioConverter alloc] initFromFormat:[input inputFormatForBus:0] toFormat:output_format];
  AVAudioFormat* input_format = [input inputFormatForBus:0];
  capture_format_ = input_format;
  [input installTapOnBus:0 bufferSize:480 format:input_format block:^(AVAudioPCMBuffer* buffer, AVAudioTime*) {
    if (buffer.floatChannelData == nullptr || converter_ == nil) return;
    if (recording_.load()) {
      NSError* write_error = nil;
      if (![native_capture_file_ writeFromBuffer:buffer error:&write_error] && write_error != nil) {
        metrics_.error = write_error.localizedDescription.UTF8String;
      }
      return;
    }
    AVAudioPCMBuffer* converted = [[AVAudioPCMBuffer alloc] initWithPCMFormat:output_format frameCapacity:512];
    __block BOOL consumed = NO;
    NSError* conversion_error = nil;
    AVAudioConverterOutputStatus conversion_status = [converter_ convertToBuffer:converted error:&conversion_error withInputFromBlock:^AVAudioBuffer*(AVAudioPacketCount, AVAudioConverterInputStatus* status) {
      if (consumed) { *status = AVAudioConverterInputStatus_EndOfStream; return nil; }
      consumed = YES; *status = AVAudioConverterInputStatus_HaveData; return buffer;
    }];
    if (conversion_status == AVAudioConverterOutputStatus_Error || conversion_error != nil) return;
    pending_audio_.insert(pending_audio_.end(), converted.floatChannelData[0],
                          converted.floatChannelData[0] + converted.frameLength);
    while (pending_audio_.size() >= 160) {
      std::vector<float> frame(pending_audio_.begin(), pending_audio_.begin() + 160);
      pending_audio_.erase(pending_audio_.begin(), pending_audio_.begin() + 160);
      const auto start = std::chrono::steady_clock::now();
      try {
        const auto enhanced = session_->process_audio(runtime::AudioFrame::mono_16k(frame),
                                      "feat_erb", "feat_spec", "m", *backend_, *pool_);
        const double latency = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        double input_energy = 0.0, output_energy = 0.0;
        for (float value : frame) input_energy += value * value;
        for (float value : enhanced.samples()) output_energy += value * value;
        metrics_.latency_ms.store(latency);
        metrics_.input_rms.store(std::sqrt(input_energy / frame.size()));
        metrics_.output_rms.store(std::sqrt(output_energy / std::max<std::size_t>(1, enhanced.sample_count())));
        metrics_.frames.fetch_add(1); pool_->reset();
        std::lock_guard<std::mutex> lock(metrics_.waveform_mutex);
        metrics_.waveform = frame;
        metrics_.output_waveform = enhanced.samples();
      } catch (const std::exception& error) { metrics_.error = error.what(); }
    }
  }];
  NSError* error = nil;
  if (![engine_ startAndReturnError:&error]) { status_label_.stringValue = error.localizedDescription; engine_ = nil; return; }
  start_button_.title = @"Stop Microphone"; status_label_.stringValue = @"Running Metal microphone pipeline";
}

- (void)refreshUI:(NSTimer*)timer {
  (void)timer;
  const double latency = metrics_.latency_ms.load();
  metrics_label_.stringValue = [NSString stringWithFormat:@"FRAMES %-6llu   LATENCY %6.2f ms   INPUT RMS %.5f   OUTPUT RMS %.5f",
                                metrics_.frames.load(), latency, metrics_.input_rms.load(), metrics_.output_rms.load()];
  [waveform_view_ setNeedsDisplay:YES];
  [output_waveform_view_ setNeedsDisplay:YES];
  if (!metrics_.error.empty()) status_label_.stringValue = [NSString stringWithUTF8String:metrics_.error.c_str()];
}
@end

int main(int argc, const char* argv[]) {
  @autoreleasepool {
    NSApplication* application = [NSApplication sharedApplication];
    AppDelegate* delegate = [[AppDelegate alloc] init];
    application.delegate = delegate;
    [application run];
  }
  return 0;
}
