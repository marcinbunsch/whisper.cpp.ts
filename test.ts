// @ts-expect-error TS(7016)
import { transcribe } from ".";

async function run() {
  const one = await transcribe();
  console.log(one);

  const two = await transcribe();
  console.log(two);
}

run();
