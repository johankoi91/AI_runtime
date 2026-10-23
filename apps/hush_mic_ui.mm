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
  std::string error;
};

}

@interface WaveformView : NSView
@property(nonatomic, assign) Metrics* metrics;
@end

@implementation WaveformView
- (void)drawRect:(NSRect)rect {
  [[NSColor colorWithRed:0.04 green:0.06 blue:0.10 alpha:1.0] setFill];
  NSRectFill(rect);
  std::vector<float> values;
  {
    std::lock_guard<std::mutex> lock(self.metrics->waveform_mutex);
    values = self.metrics->waveform;
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
  [[NSColor colorWithRed:0.25 green:0.82 blue:0.95 alpha:1.0] setStroke];
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
  NSButton* start_button_;
  NSPopUpButton* backend_popup_;
  WaveformView* waveform_view_;
  AVAudioEngine* engine_;
  Metrics metrics_;
  std::unique_ptr<runtime::Backend> backend_;
  std::unique_ptr<runtime::HushModel> model_;
  std::unique_ptr<runtime::HushStreamSession> session_;
  std::unique_ptr<runtime::MemoryPool> pool_;
}

- (void)applicationDidFinishLaunching:(NSNotification*)notification {
  (void)notification;
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

  waveform_view_ = [[WaveformView alloc] initWithFrame:NSMakeRect(32, 210, 848, 200)];
  waveform_view_.metrics = &metrics_;
  waveform_view_.wantsLayer = YES;
  waveform_view_.layer.cornerRadius = 10;
  [content addSubview:waveform_view_];
  NSTextField* wave_title = [NSTextField labelWithString:@"Live waveform / processed frame stream"];
  wave_title.textColor = [NSColor secondaryLabelColor];
  wave_title.frame = NSMakeRect(48, 380, 400, 22);
  [content addSubview:wave_title];

  metrics_label_ = [NSTextField labelWithString:@"Frames 0    Latency —    Input RMS —    Output RMS —"];
  metrics_label_.font = [NSFont monospacedSystemFontOfSize:13 weight:NSFontWeightRegular];
  metrics_label_.textColor = [NSColor colorWithRed:0.65 green:0.86 blue:0.72 alpha:1];
  metrics_label_.frame = NSMakeRect(32, 145, 848, 28);
  [content addSubview:metrics_label_];
  NSTextField* hint = [NSTextField labelWithString:@"Input: 16 kHz mono · 160 samples/frame · Metal output path · Permission required"];
  hint.textColor = [NSColor tertiaryLabelColor];
  hint.frame = NSMakeRect(32, 92, 850, 22);
  [content addSubview:hint];
  [window_ makeKeyAndOrderFront:nil];
  [NSTimer scheduledTimerWithTimeInterval:0.1 target:self selector:@selector(refreshUI:) userInfo:nil repeats:YES];
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
  [input installTapOnBus:0 bufferSize:160 format:nil block:^(AVAudioPCMBuffer* buffer, AVAudioTime*) {
    if (buffer.floatChannelData == nullptr) return;
    const float* data = buffer.floatChannelData[0];
    std::vector<float> frame(160, 0.0f);
    const std::size_t count = std::min<std::size_t>(160, buffer.frameLength);
    std::copy_n(data, count, frame.data());
    const auto start = std::chrono::steady_clock::now();
    try {
      (void)session_->process_audio(runtime::AudioFrame::mono_16k(std::move(frame)),
                                    "feat_erb", "feat_spec", "m", *backend_, *pool_);
      const double latency = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - start).count();
      metrics_.latency_ms.store(latency); metrics_.frames.fetch_add(1); pool_->reset();
      std::lock_guard<std::mutex> lock(metrics_.waveform_mutex);
      metrics_.waveform.assign(data, data + count);
    } catch (const std::exception& error) { metrics_.error = error.what(); }
  }];
  NSError* error = nil;
  if (![engine_ startAndReturnError:&error]) { status_label_.stringValue = error.localizedDescription; engine_ = nil; return; }
  start_button_.title = @"Stop Microphone"; status_label_.stringValue = @"Running Metal microphone pipeline";
}

- (void)refreshUI:(NSTimer*)timer {
  (void)timer;
  const double latency = metrics_.latency_ms.load();
  metrics_label_.stringValue = [NSString stringWithFormat:@"Frames %-6llu Latency %6.2f ms    Input RMS live    Output RMS live",
                                metrics_.frames.load(), latency];
  [waveform_view_ setNeedsDisplay:YES];
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
