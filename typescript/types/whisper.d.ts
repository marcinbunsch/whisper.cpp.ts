declare module "whisper-ts" {
  type TranscribeOptions = {
    audioData?: Float32Array;
    language?: string;
  };

  type TranscribeResult = {
    text: string;
    from: number;
    to: number;
  };

  function transcribe(options?: TranscribeOptions): Promise<TranscribeResult[]>;
}
