declare module "whisper-ts" {
  type TranscribeOptions = {
    audioData?: Float32Array;
    language?: string;
    model?: string;
  };

  type TranscribeResult = {
    text: string;
    from: number;
    to: number;
  };

  function transcribe(options?: TranscribeOptions): Promise<TranscribeResult[]>;

  class Whisper {
    constructor(modelPath: string);
    transcribe(options?: TranscribeOptions): Promise<TranscribeResult[]>;
    dispose(): void;
  }
}
