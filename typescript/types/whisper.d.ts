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

  type TranscribeWithConfidenceResult = {
    token: string;
    confidence: number;
  };

  function transcribe(options?: TranscribeOptions): Promise<TranscribeResult[]>;
  function transcribeWithConfidence(
    options?: TranscribeOptions
  ): Promise<TranscribeWithConfidenceResult[]>;

  class Whisper {
    constructor(modelPath: string);
    transcribe(options?: TranscribeOptions): Promise<TranscribeResult[]>;
    dispose(): void;
  }
}
