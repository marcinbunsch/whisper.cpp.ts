// @ts-expect-error TS(2339) I have no idea how to type a cpp module :D
const { whisper } = require("../../build/Release/whisper-ts");
const { promisify } = require("util");
const path = require("path");

const whisperAsync = promisify(whisper);

const modelsFolder = path.join(__dirname, "../../models");

const whisperParams = {
  language: "en",
  model: "ggml-base.en.bin",
};

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

module.exports = { transcribe };
