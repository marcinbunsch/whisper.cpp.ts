// @ts-expect-error TS(2339) I have no idea how to type a cpp module :D
const whisperTs = require("../../build/Release/whisper-ts");
const { promisify } = require("util");
const path = require("path");

const whisperAsync = promisify(whisperTs.whisper);

const modelsFolder = path.join(__dirname, "../../models");

const whisperParams = {
  language: "en",
  model: "ggml-base.en.bin",
};

function Whisper(model) {
  const worker = new whisperTs.WhisperWorker();
  const modelPath = path.join(modelsFolder, model);
  worker.initialize(modelPath);

  const instanceTranscribeAsync = promisify(worker.transcribe.bind(worker));
  async function instanceTransribe(options = whisperParams) {
    const params = { ...whisperParams, ...options };
    params.model = path.join(modelsFolder, params.model);
    params.audioData = options.audioData;

    const results = await instanceTranscribeAsync(params);
    const output = [];
    for (const result of results) {
      output.push({
        from: parseInt(result[0]),
        to: parseInt(result[1]),
        text: result[2].trim(),
      });
    }
    return output;
  }

  return {
    dispose() {
      worker.dispose();
    },
    async transcribe(params) {
      return instanceTransribe(params);
    },
  };
}

async function transcribe(options = whisperParams) {
  const params = { ...whisperParams, ...options };
  params.model = path.join(modelsFolder, params.model);
  params.audioData = options.audioData;

  const results = await whisperAsync(params);
  const output = [];
  for (const result of results) {
    output.push({
      from: parseInt(result[0]),
      to: parseInt(result[1]),
      text: result[2].trim(),
    });
  }
  return output;
}

module.exports = { transcribe, Whisper };
