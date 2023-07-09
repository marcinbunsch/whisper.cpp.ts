// @ts-expect-error TS(2339) I have no idea how to type a cpp module :D
const { whisper } = require("../../build/Release/whisper-ts");
const { promisify } = require("util");
const path = require("path");

const whisperAsync = promisify(whisper);

const whisperParams = {
  language: "en",
  model: path.join(__dirname, "../../models/ggml-base.en.bin"),
  fname_inp: path.join(__dirname, "../../samples/jfk.wav"),
};

async function transcribe() {
  const start = Date.now();
  const value = await whisperAsync(whisperParams);
  const elapsed = Date.now() - start;
  console.log(`transcribe() took ${elapsed}ms`);
  return value;
}

module.exports = { transcribe };
