/**
 * ElevenLabs TTS voice engine.
 * Maps characters to voice IDs and provides speak/stop functionality.
 */

const ELEVENLABS_API = "https://api.elevenlabs.io/v1/text-to-speech";

// Default voice IDs — replace with actual ElevenLabs voice IDs
const CHARACTER_VOICES = {
  Batman: "pNInz6obpgDQGcFmaJgB",   // Adam (deep, authoritative)
  Joker: "VR6AewLTigWG4xSOukaG",    // Arnold (playful, eccentric)
  Alfred: "ErXwobaYiN019PkySvjV",    // Antoni (calm, composed)
};

let currentAudio = null;

/**
 * Speak text using ElevenLabs TTS
 * Falls back to browser SpeechSynthesis if ElevenLabs fails
 */
export async function speak(text, character = "Batman", apiKey = null) {
  stop(); // Stop any currently playing audio

  // Try browser SpeechSynthesis as the primary (free) approach
  if ("speechSynthesis" in window) {
    return new Promise((resolve) => {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 0.9;
      utterance.pitch = character === "Joker" ? 1.3 : character === "Alfred" ? 0.85 : 1.0;
      utterance.volume = 0.8;

      // Try to pick a fitting voice
      const voices = speechSynthesis.getVoices();
      if (voices.length > 0) {
        const preferred = voices.find(v => v.lang.startsWith("en") && v.name.includes("Male")) || voices.find(v => v.lang.startsWith("en")) || voices[0];
        if (preferred) utterance.voice = preferred;
      }

      utterance.onend = resolve;
      utterance.onerror = resolve;
      speechSynthesis.speak(utterance);
      currentAudio = { type: "synth" };
    });
  }
}

/**
 * Speak using ElevenLabs API (premium)
 */
export async function speakWithElevenLabs(text, character = "Batman", apiKey) {
  if (!apiKey) return speak(text, character);

  stop();
  const voiceId = CHARACTER_VOICES[character] || CHARACTER_VOICES.Batman;

  try {
    const response = await fetch(`${ELEVENLABS_API}/${voiceId}`, {
      method: "POST",
      headers: {
        "xi-api-key": apiKey,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        text,
        model_id: "eleven_monolingual_v1",
        voice_settings: { stability: 0.5, similarity_boost: 0.75 },
      }),
    });

    if (!response.ok) throw new Error("ElevenLabs API error");

    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);
    currentAudio = { type: "audio", element: audio, url };

    return new Promise((resolve) => {
      audio.onended = () => { URL.revokeObjectURL(url); resolve(); };
      audio.onerror = () => { URL.revokeObjectURL(url); resolve(); };
      audio.play();
    });
  } catch (e) {
    console.warn("ElevenLabs failed, falling back to browser TTS:", e);
    return speak(text, character);
  }
}

/**
 * Stop any currently playing audio
 */
export function stop() {
  if (currentAudio?.type === "synth") {
    speechSynthesis.cancel();
  } else if (currentAudio?.type === "audio") {
    currentAudio.element.pause();
    if (currentAudio.url) URL.revokeObjectURL(currentAudio.url);
  }
  currentAudio = null;
}

export function isSpeaking() {
  if (currentAudio?.type === "synth") return speechSynthesis.speaking;
  if (currentAudio?.type === "audio") return !currentAudio.element.paused;
  return false;
}
