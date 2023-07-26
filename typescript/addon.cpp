#include "napi.h"
#include "../examples/common.h"

#include "whisper.h"

#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <cmath>
#include <cstdint>

bool DEBUG_MODE = getenv("DEBUG") != nullptr;

struct whisper_params
{
  int32_t n_threads = std::min(4, (int32_t)std::thread::hardware_concurrency());
  int32_t n_processors = 1;
  int32_t offset_t_ms = 0;
  int32_t offset_n = 0;
  int32_t duration_ms = 0;
  int32_t max_context = -1;
  int32_t max_len = 0;
  int32_t best_of = 2;
  int32_t beam_size = -1;

  float word_thold = 0.01f;
  float entropy_thold = 2.40f;
  float logprob_thold = -1.00f;

  bool speed_up = false;
  bool translate = false;
  bool diarize = false;
  bool output_txt = false;
  bool output_vtt = false;
  bool output_srt = false;
  bool output_wts = false;
  bool output_csv = false;
  bool print_special = false;
  bool print_colors = false;
  bool print_progress = false;
  bool no_timestamps = false;

  std::vector<float> audioData;

  std::string language = "en";
  std::string prompt;
  std::string model = "../../ggml-large.bin";
};

struct whisper_print_user_data
{
  const whisper_params *params;

  const std::vector<std::vector<float>> *pcmf32s;
};

//  500 -> 00:05.000
// 6000 -> 01:00.000
std::string to_timestamp(int64_t t, bool comma = false)
{
  int64_t msec = t * 10;
  int64_t hr = msec / (1000 * 60 * 60);
  msec = msec - hr * (1000 * 60 * 60);
  int64_t min = msec / (1000 * 60);
  msec = msec - min * (1000 * 60);
  int64_t sec = msec / 1000;
  msec = msec - sec * 1000;

  char buf[32];
  snprintf(buf, sizeof(buf), "%02d:%02d:%02d%s%03d", (int)hr, (int)min, (int)sec, comma ? "," : ".", (int)msec);

  return std::string(buf);
}

int timestamp_to_sample(int64_t t, int n_samples)
{
  return std::max(0, std::min((int)n_samples - 1, (int)((t * WHISPER_SAMPLE_RATE) / 100)));
}

int run(whisper_params &params, std::vector<std::vector<std::string>> &result)
{
  if (params.language != "auto" && whisper_lang_id(params.language.c_str()) == -1)
  {
    fprintf(stderr, "error: unknown language '%s'\n", params.language.c_str());
    exit(0);
  }

  // whisper init

  struct whisper_context *ctx = whisper_init_from_file(params.model.c_str());

  if (ctx == nullptr)
  {
    fprintf(stderr, "error: failed to initialize whisper context\n");
    return 3;
  }

  std::vector<float> pcmf32 = params.audioData; // mono-channel F32 PCM

  // print system information
  if (DEBUG_MODE) {
      fprintf(stderr, "\n");
      fprintf(stderr, "system_info: n_threads = %d / %d | %s\n",
              params.n_threads*params.n_processors, std::thread::hardware_concurrency(), whisper_print_system_info());
  }

  // run the inference
  {
    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

    wparams.strategy = params.beam_size > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY;

    wparams.print_realtime = false;
    wparams.print_progress = params.print_progress;
    wparams.print_timestamps = !params.no_timestamps;
    wparams.print_special = params.print_special;
    wparams.translate = params.translate;
    wparams.language = params.language.c_str();
    wparams.n_threads = params.n_threads;
    wparams.n_max_text_ctx = params.max_context >= 0 ? params.max_context : wparams.n_max_text_ctx;
    wparams.offset_ms = params.offset_t_ms;
    wparams.duration_ms = params.duration_ms;

    wparams.token_timestamps = params.output_wts || params.max_len > 0;
    wparams.thold_pt = params.word_thold;
    wparams.entropy_thold = params.entropy_thold;
    wparams.logprob_thold = params.logprob_thold;
    wparams.max_len = params.output_wts && params.max_len == 0 ? 60 : params.max_len;

    wparams.speed_up = params.speed_up;

    wparams.greedy.best_of = params.best_of;
    wparams.beam_search.beam_size = params.beam_size;

    wparams.initial_prompt = params.prompt.c_str();

    if (whisper_full_parallel(ctx, wparams, pcmf32.data(), pcmf32.size(), params.n_processors) != 0)
    {
      fprintf(stderr, "failed to process audio\n");
      return 10;
    }
  }
  // }

  const int n_segments = whisper_full_n_segments(ctx);
  result.resize(n_segments);
  for (int i = 0; i < n_segments; ++i)
  {
    const char *text = whisper_full_get_segment_text(ctx, i);
    const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
    const int64_t t1 = whisper_full_get_segment_t1(ctx, i);

    result[i].emplace_back(std::to_string(t0));
    result[i].emplace_back(std::to_string(t1));
    result[i].emplace_back(text);
  }

  // only print timings if DEBUG env var is set

  if (DEBUG_MODE) {
    whisper_print_timings(ctx);
  }

  whisper_free(ctx);

  return 0;
}

int run_with_confidence(whisper_params &params, std::vector<std::vector<std::string>> &result)
{
  if (params.language != "auto" && whisper_lang_id(params.language.c_str()) == -1)
  {
    fprintf(stderr, "error: unknown language '%s'\n", params.language.c_str());
    exit(0);
  }

  // whisper init

  struct whisper_context *ctx = whisper_init_from_file(params.model.c_str());

  if (ctx == nullptr)
  {
    fprintf(stderr, "error: failed to initialize whisper context\n");
    return 3;
  }

  std::vector<float> pcmf32 = params.audioData; // mono-channel F32 PCM

  // print system information
  if (DEBUG_MODE) {
      fprintf(stderr, "\n");
      fprintf(stderr, "system_info: n_threads = %d / %d | %s\n",
              params.n_threads*params.n_processors, std::thread::hardware_concurrency(), whisper_print_system_info());
  }

  // run the inference
  {
    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

    wparams.strategy = params.beam_size > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY;

    wparams.print_realtime = false;
    wparams.print_progress = params.print_progress;
    wparams.print_timestamps = !params.no_timestamps;
    wparams.print_special = params.print_special;
    wparams.translate = params.translate;
    wparams.language = params.language.c_str();
    wparams.n_threads = params.n_threads;
    wparams.n_max_text_ctx = params.max_context >= 0 ? params.max_context : wparams.n_max_text_ctx;
    wparams.offset_ms = params.offset_t_ms;
    wparams.duration_ms = params.duration_ms;

    wparams.token_timestamps = params.output_wts || params.max_len > 0;
    wparams.thold_pt = params.word_thold;
    wparams.entropy_thold = params.entropy_thold;
    wparams.logprob_thold = params.logprob_thold;
    wparams.max_len = params.output_wts && params.max_len == 0 ? 60 : params.max_len;

    wparams.speed_up = params.speed_up;

    wparams.greedy.best_of = params.best_of;
    wparams.beam_search.beam_size = params.beam_size;

    wparams.initial_prompt = params.prompt.c_str();

    if (whisper_full_parallel(ctx, wparams, pcmf32.data(), pcmf32.size(), params.n_processors) != 0)
    {
      fprintf(stderr, "failed to process audio\n");
      return 10;
    }
  }
  // }
  const int n_segments = whisper_full_n_segments(ctx);

  int n_tokens = 0;
  for (int i = 0; i < n_segments; i++) {
    n_tokens += whisper_full_n_tokens(ctx, i);
  }
  result.resize(n_tokens);

  int index = 0;
  for (int i = 0; i < n_segments; i++) {
    int token_count = whisper_full_n_tokens(ctx, i);
    for (int j = 0; j < token_count; ++j) {
      const char * text = whisper_full_get_token_text(ctx, i, j);
      const float  p    = whisper_full_get_token_p   (ctx, i, j);

      result[index].emplace_back(text);
      result[index].emplace_back(std::to_string(p));
      index++;
    }
  }

  // const int n_segments = whisper_full_n_segments(ctx);
  // result.resize(n_segments);
  // for (int i = 0; i < n_segments; ++i)
  // {
  //   const char *text = whisper_full_get_segment_text(ctx, i);
  //   const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
  //   const int64_t t1 = whisper_full_get_segment_t1(ctx, i);

  //   result[i].emplace_back(std::to_string(t0));
  //   result[i].emplace_back(std::to_string(t1));
  //   result[i].emplace_back(text);
  // }

  // only print timings if DEBUG env var is set

  if (DEBUG_MODE) {
    whisper_print_timings(ctx);
  }

  whisper_free(ctx);

  return 0;
};

int run_with_context(whisper_context *ctx, whisper_params &params, std::vector<std::vector<std::string>> &result)
{
  if (params.language != "auto" && whisper_lang_id(params.language.c_str()) == -1)
  {
    fprintf(stderr, "error: unknown language '%s'\n", params.language.c_str());
    exit(0);
  }

  // whisper init
  if (ctx == nullptr)
  {
    fprintf(stderr, "error: failed to initialize whisper context\n");
    return 3;
  }

  std::vector<float> pcmf32 = params.audioData; // mono-channel F32 PCM

  // print system information
  if (DEBUG_MODE) {
      fprintf(stderr, "\n");
      fprintf(stderr, "system_info: n_threads = %d / %d | %s\n",
              params.n_threads*params.n_processors, std::thread::hardware_concurrency(), whisper_print_system_info());
  }

  // run the inference
  {
    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

    wparams.strategy = params.beam_size > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY;

    wparams.print_realtime = false;
    wparams.print_progress = params.print_progress;
    wparams.print_timestamps = !params.no_timestamps;
    wparams.print_special = params.print_special;
    wparams.translate = params.translate;
    wparams.language = params.language.c_str();
    wparams.n_threads = params.n_threads;
    wparams.n_max_text_ctx = params.max_context >= 0 ? params.max_context : wparams.n_max_text_ctx;
    wparams.offset_ms = params.offset_t_ms;
    wparams.duration_ms = params.duration_ms;

    wparams.token_timestamps = params.output_wts || params.max_len > 0;
    wparams.thold_pt = params.word_thold;
    wparams.entropy_thold = params.entropy_thold;
    wparams.logprob_thold = params.logprob_thold;
    wparams.max_len = params.output_wts && params.max_len == 0 ? 60 : params.max_len;

    wparams.speed_up = params.speed_up;

    wparams.greedy.best_of = params.best_of;
    wparams.beam_search.beam_size = params.beam_size;

    wparams.initial_prompt = params.prompt.c_str();

    if (whisper_full_parallel(ctx, wparams, pcmf32.data(), pcmf32.size(), params.n_processors) != 0)
    {
      fprintf(stderr, "failed to process audio\n");
      return 10;
    }
  }
  // }

  const int n_segments = whisper_full_n_segments(ctx);
  result.resize(n_segments);
  for (int i = 0; i < n_segments; ++i)
  {
    const char *text = whisper_full_get_segment_text(ctx, i);
    const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
    const int64_t t1 = whisper_full_get_segment_t1(ctx, i);

    result[i].emplace_back(std::to_string(t0));
    result[i].emplace_back(std::to_string(t1));
    result[i].emplace_back(text);
  }

  // only print timings if DEBUG env var is set

  if (DEBUG_MODE) {
    whisper_print_timings(ctx);
  }

  return 0;
}

// class WhisperWorker {
// public:
//   const char *model_path;
//   whisper_context *ctx;

//   WhisperWorker(const char *model_path) {
//   }

//   void initialize() {
//     ctx = whisper_init_from_file(model_path);
//   }

//   void dispose() {
//     whisper_free(ctx);
//   }

//   void transcribe() {

//   }

// };

class Worker : public Napi::AsyncWorker {
public:
  Worker(Napi::Function &callback, whisper_params params)
      : Napi::AsyncWorker(callback), params(params) {}

  void Execute() override
  {
    run(params, result);
  }

  void OnOK() override
  {
    Napi::HandleScope scope(Env());
    Napi::Object res = Napi::Array::New(Env(), result.size());
    for (uint64_t i = 0; i < result.size(); ++i)
    {
      Napi::Object tmp = Napi::Array::New(Env(), 3);
      for (uint64_t j = 0; j < 3; ++j)
      {
        tmp[j] = Napi::String::New(Env(), result[i][j]);
      }
      res[i] = tmp;
    }
    Callback().Call({Env().Null(), res});
  }

private:
  whisper_params params;
  std::vector<std::vector<std::string>> result;
};

class ConfidenceWorker : public Napi::AsyncWorker {
public:
  ConfidenceWorker(Napi::Function &callback, whisper_params params)
      : Napi::AsyncWorker(callback), params(params) {}

  void Execute() override
  {
    run_with_confidence(params, result);
  }

  void OnOK() override
  {
    Napi::HandleScope scope(Env());
    Napi::Object res = Napi::Array::New(Env(), result.size());
    for (uint64_t i = 0; i < result.size(); ++i)
    {
      Napi::Object tmp = Napi::Array::New(Env(), 2);
      for (uint64_t j = 0; j < 2; ++j)
      {
        tmp[j] = Napi::String::New(Env(), result[i][j]);
      }
      res[i] = tmp;
    }
    Callback().Call({Env().Null(), res});
  }

private:
  whisper_params params;
  std::vector<std::vector<std::string>> result;
};

class WorkerWithContext : public Napi::AsyncWorker {
public:
  WorkerWithContext(Napi::Function &callback, whisper_params params, whisper_context *ctx)
      : Napi::AsyncWorker(callback), params(params), ctx(ctx) {}

  void Execute() override
  {
    run_with_context(ctx, params, result);
  }

  void OnOK() override
  {
    Napi::HandleScope scope(Env());
    Napi::Object res = Napi::Array::New(Env(), result.size());
    for (uint64_t i = 0; i < result.size(); ++i)
    {
      Napi::Object tmp = Napi::Array::New(Env(), 3);
      for (uint64_t j = 0; j < 3; ++j)
      {
        tmp[j] = Napi::String::New(Env(), result[i][j]);
      }
      res[i] = tmp;
    }
    Callback().Call({Env().Null(), res});
  }

private:
  whisper_params params;
  std::vector<std::vector<std::string>> result;
  whisper_context *ctx;
};

// class WhisperWorker : public Napi::ObjectWrap<WhisperWorker> {
//   public:
//     static Napi::Object Init(Napi::Env env, Napi::Object exports);
//     WhisperWorker(const Napi::CallbackInfo &info);
//     // Napi::Value initialize(const Napi::CallbackInfo& info);
//     // Napi::Value dispose(const Napi::CallbackInfo& info);
//     // Napi::Value transcribe(const Napi::CallbackInfo& info);

//   private:
//     whisper_context *ctx;
// };

// Napi::Object WhisperWorker::Init(Napi::Env env, Napi::Object exports) {
//     // This method is used to hook the accessor and method callbacks
//     Napi::Function func = DefineClass(env, "WhisperWorker", {
//         // InstanceMethod<&WhisperWorker::initialize>("initialize", static_cast<napi_property_attributes>(napi_writable | napi_configurable)),
//         // InstanceMethod<&WhisperWorker::dispose>("dispose", static_cast<napi_property_attributes>(napi_writable | napi_configurable)),
//         // InstanceMethod<&WhisperWorker::transcribe>("transcribe", static_cast<napi_property_attributes>(napi_writable | napi_configurable)),
//     }, nullptr);

//     Napi::FunctionReference* constructor = new Napi::FunctionReference();

//     // Create a persistent reference to the class constructor. This will allow
//     // a function called on a class prototype and a function
//     // called on instance of a class to be distinguished from each other.
//     *constructor = Napi::Persistent(func);
//     exports.Set("WhisperWorker", func);

//     // Store the constructor as the add-on instance data. This will allow this
//     // add-on to support multiple instances of itself running on multiple worker
//     // threads, as well as multiple instances of itself running in different
//     // contexts on the same thread.
//     //
//     // By default, the value set on the environment here will be destroyed when
//     // the add-on is unloaded using the `delete` operator, but it is also
//     // possible to supply a custom deleter.
//     env.SetInstanceData<Napi::FunctionReference>(constructor);

//     return exports;
// }

// Napi::Value WhisperWorker::initialize(const Napi::CallbackInfo& info){
//   const char *model_path = info[0].As<Napi::String>().Utf8Value().c_str();
//   ctx = whisper_init_from_file(model_path);
// }

// Napi::Value WhisperWorker::dispose(const Napi::CallbackInfo& info){
//   if (ctx != nullptr) {
//     whisper_free(ctx);
//   }
// }

// Napi::Value WhisperWorker::transcribe(const Napi::CallbackInfo& info){
//   Napi::Env env = info.Env();
//   if (info.Length() <= 0 || !info[0].IsObject())
//   {
//     Napi::TypeError::New(env, "object expected").ThrowAsJavaScriptException();
//   }
//   whisper_params params;

//   Napi::Object whisper_params = info[0].As<Napi::Object>();
//   std::string language = whisper_params.Get("language").As<Napi::String>();
//   std::string model = whisper_params.Get("model").As<Napi::String>();

//   std::vector<float> audioData;

//   if (whisper_params.Has("audioData"))
//   {
//     Napi::Float32Array audioDataArray = whisper_params.Get("audioData").As<Napi::Float32Array>();
//     audioData.resize(audioDataArray.ElementLength());
//     for (size_t i = 0; i < audioData.size(); i++)
//     {
//       audioData[i] = audioDataArray[i];
//     }
//   }

//   params.audioData = audioData;
//   params.language = language;
//   params.model = model;

//   Napi::Function callback = info[1].As<Napi::Function>();
//   WorkerWithContext *worker = new WorkerWithContext(callback, params, ctx);
//   worker->Queue();
//   return env.Undefined();
// }

// end of WhisperWorker

Napi::Value whisper(const Napi::CallbackInfo &info)
{
  Napi::Env env = info.Env();
  if (info.Length() <= 0 || !info[0].IsObject())
  {
    Napi::TypeError::New(env, "object expected").ThrowAsJavaScriptException();
  }
  whisper_params params;

  Napi::Object whisper_params = info[0].As<Napi::Object>();
  std::string language = whisper_params.Get("language").As<Napi::String>();
  std::string model = whisper_params.Get("model").As<Napi::String>();

  std::vector<float> audioData;

  if (whisper_params.Has("audioData"))
  {
    Napi::Float32Array audioDataArray = whisper_params.Get("audioData").As<Napi::Float32Array>();
    audioData.resize(audioDataArray.ElementLength());
    for (size_t i = 0; i < audioData.size(); i++)
    {
      audioData[i] = audioDataArray[i];
    }
  }

  params.audioData = audioData;
  params.language = language;
  params.model = model;

  Napi::Function callback = info[1].As<Napi::Function>();
  Worker *worker = new Worker(callback, params);
  worker->Queue();
  return env.Undefined();
}

Napi::Value whisperWithConfidence(const Napi::CallbackInfo &info)
{
  Napi::Env env = info.Env();
  if (info.Length() <= 0 || !info[0].IsObject())
  {
    Napi::TypeError::New(env, "object expected").ThrowAsJavaScriptException();
  }
  whisper_params params;

  Napi::Object whisper_params = info[0].As<Napi::Object>();
  std::string language = whisper_params.Get("language").As<Napi::String>();
  std::string model = whisper_params.Get("model").As<Napi::String>();

  std::vector<float> audioData;

  if (whisper_params.Has("audioData"))
  {
    Napi::Float32Array audioDataArray = whisper_params.Get("audioData").As<Napi::Float32Array>();
    audioData.resize(audioDataArray.ElementLength());
    for (size_t i = 0; i < audioData.size(); i++)
    {
      audioData[i] = audioDataArray[i];
    }
  }

  params.audioData = audioData;
  params.language = language;
  params.model = model;

  Napi::Function callback = info[1].As<Napi::Function>();
  ConfidenceWorker *worker = new ConfidenceWorker(callback, params);
  worker->Queue();
  return env.Undefined();
}

// ---------------
class WhisperWorker : public Napi::ObjectWrap<WhisperWorker> {
  public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    WhisperWorker(const Napi::CallbackInfo& info);

  private:
    Napi::Value initialize(const Napi::CallbackInfo& info);
    Napi::Value dispose(const Napi::CallbackInfo& info);
    Napi::Value transcribe(const Napi::CallbackInfo& info);
    whisper_context *ctx;
};

Napi::Object WhisperWorker::Init(Napi::Env env, Napi::Object exports) {
    // This method is used to hook the accessor and method callbacks
    Napi::Function func = DefineClass(env, "WhisperWorker", {
        InstanceMethod<&WhisperWorker::initialize>("initialize", static_cast<napi_property_attributes>(napi_writable | napi_configurable)),
        InstanceMethod<&WhisperWorker::dispose>("dispose", static_cast<napi_property_attributes>(napi_writable | napi_configurable)),
        InstanceMethod<&WhisperWorker::transcribe>("transcribe", static_cast<napi_property_attributes>(napi_writable | napi_configurable)),
    });

    Napi::FunctionReference* constructor = new Napi::FunctionReference();

    // Create a persistent reference to the class constructor. This will allow
    // a function called on a class prototype and a function
    // called on instance of a class to be distinguished from each other.
    *constructor = Napi::Persistent(func);

    exports.Set(Napi::String::New(env, "WhisperWorker"), func);

    // Store the constructor as the add-on instance data. This will allow this
    // add-on to support multiple instances of itself running on multiple worker
    // threads, as well as multiple instances of itself running in different
    // contexts on the same thread.
    //
    // By default, the value set on the environment here will be destroyed when
    // the add-on is unloaded using the `delete` operator, but it is also
    // possible to supply a custom deleter.
    env.SetInstanceData<Napi::FunctionReference>(constructor);

    return exports;
}

WhisperWorker::WhisperWorker(const Napi::CallbackInfo& info) :
    Napi::ObjectWrap<WhisperWorker>(info) {
}

Napi::Value WhisperWorker::initialize(const Napi::CallbackInfo& info){
  std::string model_path = info[0].As<Napi::String>();

  ctx = whisper_init_from_file(model_path.c_str());

  return Napi::Number::New(info.Env(), 0);
}

Napi::Value WhisperWorker::dispose(const Napi::CallbackInfo& info){
  if (ctx != nullptr) {
    if (DEBUG_MODE) {
      fprintf(stderr, "disposing whisper context\n");
    }
    whisper_free(ctx);
  }

  return Napi::Number::New(info.Env(), 0);
}

Napi::Value WhisperWorker::transcribe(const Napi::CallbackInfo& info){
  Napi::Env env = info.Env();
  if (info.Length() <= 0 || !info[0].IsObject())
  {
    Napi::TypeError::New(env, "object expected").ThrowAsJavaScriptException();
  }
  whisper_params params;

  Napi::Object whisper_params = info[0].As<Napi::Object>();
  std::string language = whisper_params.Get("language").As<Napi::String>();
  std::string model = whisper_params.Get("model").As<Napi::String>();

  std::vector<float> audioData;

  if (whisper_params.Has("audioData"))
  {
    Napi::Float32Array audioDataArray = whisper_params.Get("audioData").As<Napi::Float32Array>();
    audioData.resize(audioDataArray.ElementLength());
    for (size_t i = 0; i < audioData.size(); i++)
    {
      audioData[i] = audioDataArray[i];
    }
  }

  params.audioData = audioData;
  params.language = language;
  params.model = model;

  Napi::Function callback = info[1].As<Napi::Function>();
  WorkerWithContext *worker = new WorkerWithContext(callback, params, ctx);
  worker->Queue();
  return env.Undefined();
}

// --------------

Napi::Object Init(Napi::Env env, Napi::Object exports)
{
  exports.Set(
    Napi::String::New(env, "whisper"),
    Napi::Function::New(env, whisper)
  );

  exports.Set(
    Napi::String::New(env, "whisperWithConfidence"),
    Napi::Function::New(env, whisperWithConfidence)
  );

  // WhisperWorker::Init(env, exports);
  WhisperWorker::Init(env, exports);

  return exports;
}

NODE_API_MODULE(whisper, Init);
