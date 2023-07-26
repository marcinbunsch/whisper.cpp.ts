import { transcribeWithConfidence } from "whisper-ts";
import nodeWav from "node-wav";
import fs from "fs";

function loadAudio(audioPath: string) {
  const buffer = fs.readFileSync(audioPath);
  const result = nodeWav.decode(buffer);
  const audioData = new Float32Array(result.channelData[0].length);
  for (let i = 0; i < audioData.length; i++) {
    for (let j = 0; j < result.channelData.length; j++) {
      audioData[i] += result.channelData[j][i];
    }
  }
  return { audioData, sampleRate: result.sampleRate };
}

async function run() {
  const filename = "jfk.wav";
  const { audioData } = loadAudio(filename);

  const start = Date.now();
  const two = await transcribeWithConfidence({
    audioData,
  });
  console.log(two);
  const elapsed = Date.now() - start;
  console.log(`transcribe() took ${elapsed}ms`);
}

run();
