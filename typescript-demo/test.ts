import { transcribe } from "whisper-ts";

async function run() {
  const one = await transcribe();
  console.log(one);

  const two = await transcribe();
  console.log(two);
}

run();
